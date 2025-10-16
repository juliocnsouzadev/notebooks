"""
Utility functions for the Agentic AI Chart Generation notebook.
This module provides helper functions for data loading, LLM interactions,
image processing, and HTML display.
"""

import os
import base64
import pandas as pd
from datetime import datetime
from typing import Union, Tuple
from IPython.display import display, HTML, Image
from openai import OpenAI
import anthropic


# Initialize API clients
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load coffee sales data from CSV and add derived features.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with original columns plus year, quarter, and month features
    """
    df = pd.read_csv(csv_path)
    
    # Parse date column if it exists
    if 'date' in df.columns:
        # Try different date formats
        try:
            df['date_parsed'] = pd.to_datetime(df['date'], format='%m/%d/%y')
        except:
            try:
                df['date_parsed'] = pd.to_datetime(df['date'])
            except:
                print("Warning: Could not parse date column")
                return df
        
        # Extract year, quarter, and month
        df['year'] = df['date_parsed'].dt.year
        df['quarter'] = df['date_parsed'].dt.quarter
        df['month'] = df['date_parsed'].dt.month
    
    return df


def print_html(content: Union[str, pd.DataFrame], 
               title: str = None, 
               is_image: bool = False):
    """
    Display content as HTML in Jupyter notebook.
    
    Args:
        content: String, DataFrame, or image path to display
        title: Optional title to display above the content
        is_image: If True, treat content as an image path
    """
    html_parts = []
    
    # Add title if provided
    if title:
        html_parts.append(f"""
        <div style="border-left: 4px solid #3b82f6; padding: 12px; margin: 16px 0; 
                    background: #eff6ff; border-radius: 4px;">
            <h3 style="margin: 0; color: #1e40af; font-family: system-ui, -apple-system, sans-serif;">
                {title}
            </h3>
        </div>
        """)
    
    # Handle different content types
    if is_image:
        # Display image
        if os.path.exists(content):
            display(Image(filename=content))
        else:
            html_parts.append(f"""
            <div style="padding: 12px; color: #dc2626; background: #fee; border-radius: 4px;">
                ⚠️ Image file not found: {content}
            </div>
            """)
            display(HTML(''.join(html_parts)))
        return
    
    elif isinstance(content, pd.DataFrame):
        # Display DataFrame
        html_parts.append(content.to_html(index=False))
    
    else:
        # Display text content with code formatting if it looks like code
        if '<execute_python>' in str(content) or 'import' in str(content) or 'def ' in str(content):
            # Format as code
            html_parts.append(f"""
            <div style="background: #f8f9fa; border: 1px solid #dee2e6; 
                        border-radius: 4px; padding: 16px; margin: 8px 0;">
                <pre style="margin: 0; font-family: 'Courier New', monospace; 
                           white-space: pre-wrap; word-wrap: break-word;">{content}</pre>
            </div>
            """)
        else:
            # Format as regular text
            html_parts.append(f"""
            <div style="padding: 12px; margin: 8px 0; line-height: 1.6; 
                        font-family: system-ui, -apple-system, sans-serif;">
                {content}
            </div>
            """)
    
    display(HTML(''.join(html_parts)))


def get_response(model: str, prompt: str) -> str:
    """
    Get a response from an LLM (OpenAI or Anthropic).
    
    Args:
        model: Model name (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet-20241022')
        prompt: The prompt to send to the model
        
    Returns:
        The model's response as a string
    """
    model_lower = model.lower()
    
    # Determine which API to use based on model name
    if 'gpt' in model_lower or 'o1' in model_lower or 'o3' in model_lower or 'o4' in model_lower:
        # OpenAI API
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    elif 'claude' in model_lower or 'anthropic' in model_lower:
        # Anthropic API
        response = anthropic_client.messages.create(
            model=model if 'claude' in model else "claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    else:
        raise ValueError(f"Unsupported model: {model}. Use 'gpt-*', 'o1-*', 'o3-*', 'o4-*' or 'claude-*'")


def encode_image_b64(image_path: str) -> Tuple[str, str]:
    """
    Encode an image file to base64.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (media_type, base64_encoded_string)
    """
    # Determine media type from file extension
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_types.get(ext, 'image/png')
    
    # Read and encode the image
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    return media_type, encoded_string


def image_openai_call(model_name: str, prompt: str, media_type: str, b64_data: str) -> str:
    """
    Call OpenAI API with an image (vision model).
    
    Args:
        model_name: OpenAI model name (e.g., 'gpt-4o', 'gpt-4o-mini')
        prompt: Text prompt
        media_type: Image media type (e.g., 'image/png')
        b64_data: Base64 encoded image data
        
    Returns:
        The model's response as a string
    """
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64_data}"
                        }
                    }
                ]
            }
        ],
        max_tokens=4096,
        temperature=0.7
    )
    
    return response.choices[0].message.content


def image_anthropic_call(model_name: str, prompt: str, media_type: str, b64_data: str) -> str:
    """
    Call Anthropic API with an image (Claude vision model).
    
    Args:
        model_name: Anthropic model name (e.g., 'claude-3-5-sonnet-20241022')
        prompt: Text prompt
        media_type: Image media type (e.g., 'image/png')
        b64_data: Base64 encoded image data
        
    Returns:
        The model's response as a string
    """
    # Use the model name if it contains 'claude', otherwise default
    if 'claude' not in model_name.lower():
        model_name = "claude-3-5-sonnet-20241022"
    
    response = anthropic_client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    # Join all text blocks from the response
    return ''.join([block.text for block in response.content if hasattr(block, 'text')])


def ensure_execute_python_tags(code: str) -> str:
    """
    Ensure code is wrapped in <execute_python> tags.
    
    Args:
        code: Python code string
        
    Returns:
        Code wrapped in <execute_python> tags
    """
    if '<execute_python>' not in code:
        return f"<execute_python>\n{code}\n</execute_python>"
    return code

