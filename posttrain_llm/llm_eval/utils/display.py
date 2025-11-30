"""
Display utilities for formatted output.
"""


def display_section_header(title: str, level: int = 1):
    """
    Display a section header with appropriate formatting.
    
    Args:
        title: Section title
        level: Header level (1, 2, or 3)
            - Level 1: Major section with double lines
            - Level 2: Subsection with single lines
            - Level 3: Minor section with dots
            
    Example:
        >>> display_section_header("Main Section", level=1)
        ============================================================
        MAIN SECTION
        ============================================================
        
        >>> display_section_header("Subsection", level=2)
        ----------------------------------------
        Subsection
        ----------------------------------------
    """
    if level == 1:
        print(f"\n{'='*60}")
        print(f"{title.upper()}")
        print('='*60)
    elif level == 2:
        print(f"\n{'-'*40}")
        print(f"{title}")
        print('-'*40)
    else:
        print(f"\n{title}")
        print('·'*len(title))


def display_warning(message: str):
    """
    Display a warning message in a prominent way.
    
    Args:
        message: Warning message to display
        
    Example:
        >>> display_warning("Model not found")
        ⚠️  WARNING: Model not found
    """
    print("⚠️  WARNING:", message)


def display_success(message: str):
    """
    Display a success message.
    
    Args:
        message: Success message to display
        
    Example:
        >>> display_success("Model loaded successfully")
        ✅ Model loaded successfully
    """
    print("✅", message)


def display_info(message: str):
    """
    Display an info message.
    
    Args:
        message: Info message to display
        
    Example:
        >>> display_info("Loading dataset...")
        ℹ️  Loading dataset...
    """
    print("ℹ️ ", message)


def display_error(message: str):
    """
    Display an error message.
    
    Args:
        message: Error message to display
        
    Example:
        >>> display_error("Failed to load model")
        ❌ Failed to load model
    """
    print("❌", message)
