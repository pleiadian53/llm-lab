"""
HuggingFace utilities for authentication and model access.
"""

import os
from huggingface_hub import HfApi


def validate_token() -> bool:
    """
    Validate the current Hugging Face token and display user information.
    
    This function should be called after setting up your token to ensure
    it was configured correctly.
    
    Returns:
        bool: True if token is valid, False otherwise
        
    Example:
        >>> if validate_token():
        ...     print("Ready to download models!")
        ... else:
        ...     print("Please set up your HuggingFace token")
    """
    try:
        # Try to get token from environment variable first
        token = None
        for env_var in ['HUGGINGFACE_HUB_TOKEN', 'HF_TOKEN']:
            token = os.environ.get(env_var)
            if token:
                break
        
        if token:
            # Set the token for the API
            api = HfApi(token=token)
        else:
            # Try without explicit token (maybe already logged in)
            api = HfApi()
            
        user_info = api.whoami()
        print(f"✅ Token validated successfully!")
        print(f"   Logged in as: {user_info['name']}")
        print(f"   Token type: {user_info.get('type', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Token validation failed.")
        print(f"   Error: {e}")
        print("\n💡 Please check that:")
        print("   1. Your token is correctly set")
        print("   2. Your token has the necessary permissions") 
        print("   3. You have access to the models you need")
        print("   4. Try running: huggingface-cli login")
        return False
