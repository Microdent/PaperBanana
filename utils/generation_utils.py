# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
import re
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import httpx
import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}


def load_model_config():
    """Reload model config from disk so runtime credential updates are visible."""
    global model_config
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8-sig") as f:
            model_config = yaml.safe_load(f) or {}
    else:
        model_config = {}
    return model_config


load_model_config()

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default


def get_vertex_ai_config():
    """Return Vertex AI project/location from env or config."""
    project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
    location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
    return project_id, location


def get_api_base_url(provider: str) -> str:
    """Return optional custom base URL for a provider."""
    provider_map = {
        "google": ("api_base_urls", "google_genai_base_url", "GOOGLE_GENAI_BASE_URL"),
        "openai": ("api_base_urls", "openai_base_url", "OPENAI_BASE_URL"),
        "anthropic": ("api_base_urls", "anthropic_base_url", "ANTHROPIC_BASE_URL"),
    }
    section, key, env_var = provider_map[provider]
    return get_config_val(section, key, env_var, "")


def get_google_http_options():
    """Build google-genai HttpOptions when a custom base URL is configured."""
    google_base_url = get_api_base_url("google")
    if not google_base_url:
        return None, ""

    http_options = types.HttpOptionsDict(
        base_url=google_base_url,
        base_url_resource_scope=types.ResourceScope.COLLECTION,
    )
    return http_options, google_base_url


def initialize_gemini_client():
    """Initialize Gemini using either Google API key or Vertex AI credentials."""
    http_options, google_base_url = get_google_http_options()
    base_url_suffix = f" via custom base URL ({google_base_url})" if google_base_url else ""

    api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
    if api_key:
        client_kwargs = {"api_key": api_key}
        if http_options is not None:
            client_kwargs["http_options"] = http_options
        return genai.Client(**client_kwargs), f"Gemini (Google API key){base_url_suffix}"

    project_id, location = get_vertex_ai_config()
    if project_id:
        client_kwargs = {"vertexai": True, "project": project_id, "location": location}
        if http_options is not None:
            client_kwargs["http_options"] = http_options
        return (
            genai.Client(**client_kwargs),
            f"Gemini (Vertex AI: {project_id}/{location}){base_url_suffix}",
        )

    return None, ""

# Initialize clients lazily or with robust defaults
gemini_client = None
anthropic_client = None
openai_client = None
openrouter_client = None
openai_api_key = ""
openai_base_url = ""
openrouter_api_key = ""


def reinitialize_clients():
    """(Re)build all API clients from current env vars / config file.

    Called once at module load and can be called again at runtime
    (e.g. after the user sets new API keys via the Gradio UI).

    Returns a list of client names that were successfully initialized.
    """
    global gemini_client, anthropic_client, openai_client
    global openrouter_client, openrouter_api_key
    global openai_api_key, openai_base_url

    initialized = []

    load_model_config()

    gemini_client, gemini_label = initialize_gemini_client()
    if gemini_client is not None:
        print(f"Initialized {gemini_label}")
        initialized.append(gemini_label)

    key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
    if key:
        anthropic_kwargs = {"api_key": key}
        anthropic_base_url = get_api_base_url("anthropic")
        if anthropic_base_url:
            anthropic_kwargs["base_url"] = anthropic_base_url
        anthropic_client = AsyncAnthropic(**anthropic_kwargs)
        label = "Anthropic"
        if anthropic_base_url:
            label += f" (base URL: {anthropic_base_url})"
        print(f"Initialized {label}")
        initialized.append(label)
    else:
        anthropic_client = None

    openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
    openai_base_url = get_api_base_url("openai")
    if openai_api_key:
        openai_kwargs = {"api_key": openai_api_key}
        if openai_base_url:
            openai_kwargs["base_url"] = openai_base_url
        openai_client = AsyncOpenAI(**openai_kwargs)
        label = "OpenAI"
        if openai_base_url:
            label += f" (base URL: {openai_base_url})"
        print(f"Initialized {label}")
        initialized.append(label)
    else:
        openai_client = None

    openrouter_api_key = get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")
    if openrouter_api_key:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        print("Initialized OpenRouter Client with API Key")
        initialized.append("OpenRouter")
    else:
        openrouter_client = None

    return initialized


# Run once at import time (preserves original behaviour)
reinitialize_clients()



def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["image_base64"]),
                        mime_type="image/jpeg",
                    )
                )
            elif "data" in item:
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(item["data"]),
                        mime_type=item.get("mime_type", "image/jpeg"),
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client was not initialized: missing Google API key or Vertex AI project configuration. "
            "Please set GOOGLE_API_KEY, or configure GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION "
            "(or google_cloud.project_id / google_cloud.location in configs/model_config.yaml)."
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client

            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if (
                "nanoviz" in model_name
                or "image" in model_name
            ):
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                    if part.text is not None
                ]
            result_list.extend([r for r in raw_response_list if r and r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            
            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2 ** attempt), 30)
            
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    
    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    
    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            elif "image_base64" in item:
                # Shorthand format used by planner_agent
                data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            elif "data" in item:
                media_type = item.get("mime_type", "image/jpeg")
                data = item.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            content = first_response.choices[0].message.content or ""
            if not content.strip():
                print(f"OpenAI returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_openrouter_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter API (OpenAI-compatible) with asynchronous retry logic.
    """
    if openrouter_client is None:
        raise RuntimeError(
            "OpenRouter client was not initialized: missing API key. "
            "Please set OPENROUTER_API_KEY in environment, or configure "
            "api_keys.openrouter_api_key in configs/model_config.yaml."
        )

    model_name = _to_openrouter_model_id(model_name)
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents

    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            content = first_response.choices[0].message.content or ""
            if not content.strip():
                print(f"OpenRouter returned empty content, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
            response_text_list.append(content)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"OpenRouter attempt {attempt + 1} failed{context_msg}: {error_str}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)

    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        print(f"Error: All {max_attempts} OpenRouter attempts failed{context_msg}.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents},
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent OpenRouter candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content or "Error")

    return response_text_list


async def call_openrouter_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenRouter image generation via direct httpx POST to avoid
    openai SDK issues with extra_body dropping the model field.
    Images are returned in choices[0].message.content as inline_data or
    in choices[0].message.images as data URLs.
    """
    if not openrouter_api_key:
        raise RuntimeError(
            "OpenRouter client was not initialized: missing API key."
        )

    system_prompt = config.get("system_prompt", "")
    temperature = config.get("temperature", 1.0)
    aspect_ratio = config.get("aspect_ratio", "1:1")
    image_size = config.get("image_size", "1k")

    model_name = _to_openrouter_model_id(model_name)
    openai_contents = _convert_to_openai_format(contents)

    image_config = {}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config["image_size"] = image_size

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": openai_contents},
        ],
        "temperature": temperature,
        "modalities": ["image", "text"],
    }
    if image_config:
        payload["image_config"] = image_config

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()

            choices = data.get("choices", [])
            if not choices:
                print(f"[Warning]: OpenRouter image generation returned no choices, retrying...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

            message = choices[0].get("message", {})

            # Try extracting from inline_data in content (Gemini-style)
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "inline_data" in part:
                        b64_data = part["inline_data"].get("data", "")
                        if b64_data:
                            return [b64_data]

            # Try extracting from images field (OpenRouter standard)
            images = message.get("images")
            if images and len(images) > 0:
                img_item = images[0]
                if isinstance(img_item, dict):
                    data_url = img_item.get("image_url", {}).get("url", "")
                else:
                    data_url = str(img_item)
                if "," in data_url:
                    b64_data = data_url.split(",", 1)[1]
                else:
                    b64_data = data_url
                if b64_data:
                    return [b64_data]

            # Try extracting base64 from text content
            if isinstance(content, str) and content.startswith("data:image"):
                if "," in content:
                    b64_data = content.split(",", 1)[1]
                    if b64_data:
                        return [b64_data]

            print(f"[Warning]: OpenRouter image generation returned no images, retrying...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            continue

        except httpx.HTTPStatusError as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"OpenRouter image gen attempt {attempt + 1} failed{context_msg}: "
                f"HTTP {e.response.status_code} - {e.response.text}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"OpenRouter image gen attempt {attempt + 1} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


def _strip_known_provider_prefix(model_name: str) -> tuple[str | None, str]:
    """Strip explicit provider prefixes like openai/ or openrouter/."""
    for provider in ("openrouter", "openai", "anthropic", "gemini"):
        prefix = f"{provider}/"
        if model_name.startswith(prefix):
            return provider, model_name[len(prefix):]
    return None, model_name


def _to_openrouter_model_id(model_name: str) -> str:
    """Convert a bare model name to OpenRouter format (provider/model).

    OpenRouter requires model IDs like 'google/gemini-3-pro-preview'.
    If the name already contains '/', assume it's already qualified.
    Otherwise, prefix with 'google/' for Gemini models.
    """
    if "/" in model_name:
        return model_name
    if model_name.startswith("gemini"):
        return f"google/{model_name}"
    return model_name


def _resolve_provider_and_model_name(model_name: str) -> tuple[str, str]:
    """Resolve provider routing for text/image calls."""
    explicit_provider, stripped_model_name = _strip_known_provider_prefix(model_name)

    if explicit_provider == "openrouter":
        return "openrouter", stripped_model_name
    if explicit_provider == "openai":
        return "openai", stripped_model_name
    if explicit_provider == "anthropic":
        return "anthropic", stripped_model_name
    if explicit_provider == "gemini":
        return "gemini", stripped_model_name

    if stripped_model_name.startswith("claude-"):
        return "anthropic", stripped_model_name
    if any(stripped_model_name.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
        return "openai", stripped_model_name

    if openrouter_client is not None:
        return "openrouter", _to_openrouter_model_id(stripped_model_name)
    if gemini_client is not None:
        return "gemini", stripped_model_name
    if anthropic_client is not None:
        return "anthropic", stripped_model_name
    if openai_client is not None:
        return "openai", stripped_model_name

    raise RuntimeError(
        "No API client available. Please configure at least one credential source "
        "(OpenRouter, Gemini via GOOGLE_API_KEY, Gemini via Vertex AI / GOOGLE_CLOUD_PROJECT, "
        "Anthropic, or OpenAI) in configs/model_config.yaml or via environment variables."
    )


def _looks_like_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _extract_url_from_markdown(value: str) -> str:
    match = re.search(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", value)
    return match.group(1) if match else ""


def _extract_data_url_b64(value: str) -> str:
    if value.startswith("data:image"):
        return value.split(",", 1)[1] if "," in value else value
    return ""


def _extract_first_image_ref_from_obj(obj: Any) -> tuple[str, str]:
    """Return ('b64'|'url'|'', value) from many OpenAI-compatible/New API shapes."""
    if obj is None:
        return "", ""

    if isinstance(obj, str):
        value = obj.strip()
        if not value:
            return "", ""

        data_b64 = _extract_data_url_b64(value)
        if data_b64:
            return "b64", data_b64

        if _looks_like_http_url(value):
            return "url", value

        md_url = _extract_url_from_markdown(value)
        if md_url:
            return "url", md_url

        normalized = value
        if value.startswith("```") and value.endswith("```"):
            normalized = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", value)
            normalized = re.sub(r"\n?```$", "", normalized).strip()

        if normalized.startswith("{") or normalized.startswith("["):
            try:
                parsed = json.loads(normalized)
            except Exception:
                return "", ""
            return _extract_first_image_ref_from_obj(parsed)

        return "", ""

    if isinstance(obj, list):
        for item in obj:
            ref_type, ref_value = _extract_first_image_ref_from_obj(item)
            if ref_type:
                return ref_type, ref_value
        return "", ""

    if isinstance(obj, dict):
        if isinstance(obj.get("b64_json"), str) and obj.get("b64_json"):
            return "b64", obj["b64_json"]

        inline_data = obj.get("inline_data")
        if isinstance(inline_data, dict):
            data_value = inline_data.get("data", "")
            if isinstance(data_value, str) and data_value:
                return "b64", data_value

        for key_path in (
            ("image_url", "url"),
            ("file", "url"),
        ):
            nested = obj
            valid = True
            for key in key_path:
                if isinstance(nested, dict) and key in nested:
                    nested = nested[key]
                else:
                    valid = False
                    break
            if valid:
                ref_type, ref_value = _extract_first_image_ref_from_obj(nested)
                if ref_type:
                    return ref_type, ref_value

        for key in (
            "url",
            "uri",
            "text",
            "content",
            "images",
            "image",
            "output",
            "data",
            "message",
            "choices",
            "candidates",
            "parts",
            "result",
        ):
            if key in obj:
                ref_type, ref_value = _extract_first_image_ref_from_obj(obj[key])
                if ref_type:
                    return ref_type, ref_value

        for value in obj.values():
            ref_type, ref_value = _extract_first_image_ref_from_obj(value)
            if ref_type:
                return ref_type, ref_value

    return "", ""


async def _download_image_url_as_b64(url: str) -> str:
    """Download an image URL and return base64 bytes."""
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        resp = await client.get(url)
    resp.raise_for_status()

    content_type = (resp.headers.get("content-type") or "").lower()
    data = resp.content
    if content_type.startswith("image/"):
        return base64.b64encode(data).decode("utf-8")

    if data.startswith(b"\x89PNG") or data.startswith(b"\xff\xd8\xff") or data.startswith(b"GIF87a") or data.startswith(b"GIF89a") or data.startswith(b"RIFF"):
        return base64.b64encode(data).decode("utf-8")

    return ""


async def extract_first_b64_image_from_openai_compatible_response(data: Dict[str, Any]) -> str:
    """Extract base64 image payload from many OpenAI-compatible/New API response formats."""
    ref_type, ref_value = _extract_first_image_ref_from_obj(data)
    if ref_type == "b64":
        return ref_value
    if ref_type == "url":
        try:
            return await _download_image_url_as_b64(ref_value)
        except Exception as e:
            print(f"[Warning]: failed to download generated image URL: {e}")
            return ""
    return ""


def _contents_have_image_input(contents: List[Dict[str, Any]]) -> bool:
    return any(item.get("type") == "image" for item in contents)


def _extract_text_prompt(contents: List[Dict[str, Any]]) -> str:
    return "\n".join(item.get("text", "") for item in contents if item.get("type") == "text").strip()


async def call_openai_compatible_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context="", provider_label="OpenAI-compatible"
):
    """Call an OpenAI-compatible /chat/completions endpoint for image generation."""
    if not openai_api_key:
        raise RuntimeError(
            "OpenAI-compatible client was not initialized: missing OPENAI_API_KEY / api_keys.openai_api_key."
        )

    base_url = (openai_base_url or "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    system_prompt = config.get("system_prompt", "")
    temperature = config.get("temperature", 1.0)
    aspect_ratio = config.get("aspect_ratio", "1:1")
    image_size = config.get("image_size", "1k")

    openai_contents = _convert_to_openai_format(contents)

    image_config = {}
    if aspect_ratio:
        image_config["aspect_ratio"] = aspect_ratio
    if image_size:
        image_config["image_size"] = image_size

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": openai_contents},
        ],
        "temperature": temperature,
        "modalities": ["image", "text"],
    }
    if image_config:
        payload["image_config"] = image_config

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            b64_data = await extract_first_b64_image_from_openai_compatible_response(data)
            if b64_data:
                return [b64_data]

            choices = data.get("choices", [])
            message_keys = []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    message_keys = list(message.keys())
            print(
                f"[Warning]: {provider_label} image generation returned no images, "
                f"retrying... top-level keys={list(data.keys())}, message keys={message_keys}"
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            continue

        except httpx.HTTPStatusError as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"{provider_label} image gen attempt {attempt + 1} failed{context_msg}: "
                f"HTTP {e.response.status_code} - {e.response.text}. Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"{provider_label} image gen attempt {attempt + 1} failed{context_msg}: {e}. "
                f"Retrying in {current_delay}s..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_image_generation_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """Unified image-generation router for Gemini/OpenAI-compatible/OpenRouter."""
    provider, actual_model = _resolve_provider_and_model_name(model_name)

    if provider == "openai":
        if actual_model.startswith("gpt-image") and not _contents_have_image_input(contents):
            prompt = _extract_text_prompt(contents)
            if not prompt:
                raise RuntimeError("OpenAI GPT-Image generation requires a text prompt.")
            return await call_openai_image_generation_with_retry_async(
                model_name=actual_model,
                prompt=prompt,
                config=config,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                error_context=error_context,
            )
        return await call_openai_compatible_image_generation_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
            provider_label="OpenAI-compatible",
        )

    if provider == "openrouter":
        return await call_openrouter_image_generation_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    if provider == "gemini":
        return await call_gemini_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=config.get("system_prompt", ""),
                temperature=config.get("temperature", 1.0),
                candidate_count=config.get("candidate_count", 1),
                max_output_tokens=config.get("max_output_tokens", 50000),
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=config.get("aspect_ratio", "1:1"),
                    image_size=config.get("image_size", "1k"),
                ),
            ),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    raise RuntimeError(
        f"Image generation is not supported for provider '{provider}' with model '{actual_model}'."
    )


async def call_model_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    Unified router that dispatches to the correct provider based on model_name.

    Routing rules:
      1. Explicit prefix overrides: "openrouter/", "openai/", "anthropic/", "gemini/"
      2. Otherwise auto-detect based on configured credentials.
    """
    provider, actual_model = _resolve_provider_and_model_name(model_name)

    if provider == "gemini":
        return await call_gemini_with_retry_async(
            model_name=actual_model,
            contents=contents,
            config=config,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )

    # Convert Gemini GenerateContentConfig -> dict for OpenAI/Claude/OpenRouter
    cfg_dict = {
        "system_prompt": config.system_instruction if hasattr(config, "system_instruction") else "",
        "temperature": config.temperature if hasattr(config, "temperature") else 1.0,
        "candidate_num": config.candidate_count if hasattr(config, "candidate_count") else 1,
        "max_completion_tokens": config.max_output_tokens if hasattr(config, "max_output_tokens") else 50000,
    }

    call_fn = {
        "openrouter": call_openrouter_with_retry_async,
        "anthropic": call_claude_with_retry_async,
        "openai": call_openai_with_retry_async,
    }[provider]

    return await call_fn(
        model_name=actual_model,
        contents=contents,
        config=cfg_dict,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
    )
