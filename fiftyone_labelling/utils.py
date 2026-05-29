# save as: utils.py

"""
Utility functions for hashing and timestamps

Common utilities for generating hashes, timestamps, and other helper functions.

Usage:
    from utils import generate_hash, get_timestamp, create_unique_id
    
    # Generate hash
    hash_str = generate_hash(length=8)
    
    # Get timestamp
    timestamp = get_timestamp()
    
    # Create unique ID
    unique_id = create_unique_id(prefix="img")
"""

import hashlib
import secrets
import string
import uuid
from datetime import datetime
from typing import Optional, Union
import random


# ============================================================================
# HASH GENERATION FUNCTIONS
# ============================================================================

def generate_hash(length: int = 8, method: str = "hex") -> str:
    """
    Generate a random hash of specified length
    
    Args:
        length: Length of the hash string (default: 8)
        method: Hash generation method
            - "hex": Hexadecimal characters (0-9, a-f)
            - "alphanumeric": Letters and numbers (a-z, A-Z, 0-9)
            - "lowercase": Lowercase letters and numbers (a-z, 0-9)
            - "base64": Base64-like characters
            - "uuid": UUID-based (length parameter ignored)
            
    Returns:
        Random hash string of specified length
        
    Examples:
        >>> generate_hash(8, "hex")
        'a3f7b2c9'
        
        >>> generate_hash(12, "alphanumeric")
        'K3mP9xQz4RtY'
        
        >>> generate_hash(16, "lowercase")
        'j4k2m9p1q5r8s3t7'
    """
    
    if method == "hex":
        # Generate random hexadecimal string
        return secrets.token_hex(length // 2 + 1)[:length]
    
    elif method == "alphanumeric":
        # Generate alphanumeric (case-sensitive)
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    elif method == "lowercase":
        # Generate lowercase alphanumeric
        alphabet = string.ascii_lowercase + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    elif method == "base64":
        # Generate URL-safe base64-like string
        return secrets.token_urlsafe(length)[:length]
    
    elif method == "uuid":
        # Generate UUID (length parameter ignored)
        return str(uuid.uuid4())
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hex', 'alphanumeric', 'lowercase', 'base64', or 'uuid'")


def generate_md5_hash(data: Union[str, bytes], length: Optional[int] = None) -> str:
    """
    Generate MD5 hash from data
    
    Args:
        data: String or bytes to hash
        length: Optional length to truncate hash (default: full 32 characters)
        
    Returns:
        MD5 hash string
        
    Examples:
        >>> generate_md5_hash("hello world")
        '5eb63bbbe01eeed093cb22bb8f5acdc3'
        
        >>> generate_md5_hash("hello world", length=8)
        '5eb63bbb'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hash_obj = hashlib.md5(data)
    hash_str = hash_obj.hexdigest()
    
    if length:
        return hash_str[:length]
    return hash_str


def generate_sha256_hash(data: Union[str, bytes], length: Optional[int] = None) -> str:
    """
    Generate SHA256 hash from data
    
    Args:
        data: String or bytes to hash
        length: Optional length to truncate hash (default: full 64 characters)
        
    Returns:
        SHA256 hash string
        
    Examples:
        >>> generate_sha256_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        
        >>> generate_sha256_hash("hello world", length=16)
        'b94d27b9934d3e08'
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hash_obj = hashlib.sha256(data)
    hash_str = hash_obj.hexdigest()
    
    if length:
        return hash_str[:length]
    return hash_str


def hash_file(filepath: str, algorithm: str = "md5", length: Optional[int] = None) -> str:
    """
    Generate hash of a file
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', 'sha1')
        length: Optional length to truncate hash
        
    Returns:
        Hash string of file contents
        
    Examples:
        >>> hash_file("image.jpg", "md5")
        'a3f7b2c9e5d1f6a8b4c7e2d9f3a6b8c1'
    """
    if algorithm == "md5":
        hash_obj = hashlib.md5()
    elif algorithm == "sha256":
        hash_obj = hashlib.sha256()
    elif algorithm == "sha1":
        hash_obj = hashlib.sha1()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    hash_str = hash_obj.hexdigest()
    
    if length:
        return hash_str[:length]
    return hash_str


# ============================================================================
# TIMESTAMP FUNCTIONS
# ============================================================================

def get_timestamp(format: str = "default", separator: str = "-") -> str:
    """
    Get current timestamp in various formats
    
    Args:
        format: Timestamp format
            - "default": YYYY-MM-DD-HH-MM (year-month-day-hour-minute)
            - "full": YYYY-MM-DD-HH-MM-SS (includes seconds)
            - "date": YYYY-MM-DD (date only)
            - "time": HH-MM-SS (time only)
            - "compact": YYYYMMDDHHMMSS (no separators)
            - "iso": ISO 8601 format (YYYY-MM-DDTHH:MM:SS)
            - "unix": Unix timestamp (seconds since epoch)
        separator: Character to use as separator (default: "-")
        
    Returns:
        Formatted timestamp string
        
    Examples:
        >>> get_timestamp()
        '2024-01-15-14-30'
        
        >>> get_timestamp("full")
        '2024-01-15-14-30-45'
        
        >>> get_timestamp("date")
        '2024-01-15'
        
        >>> get_timestamp("compact")
        '20240115143045'
        
        >>> get_timestamp("unix")
        '1705330245'
    """
    now = datetime.now()
    
    if format == "default":
        return now.strftime(f"%Y{separator}%m{separator}%d{separator}%H{separator}%M")
    
    elif format == "full":
        return now.strftime(f"%Y{separator}%m{separator}%d{separator}%H{separator}%M{separator}%S")
    
    elif format == "date":
        return now.strftime(f"%Y{separator}%m{separator}%d")
    
    elif format == "time":
        return now.strftime(f"%H{separator}%M{separator}%S")
    
    elif format == "compact":
        return now.strftime("%Y%m%d%H%M%S")
    
    elif format == "iso":
        return now.isoformat()
    
    elif format == "unix":
        return str(int(now.timestamp()))
    
    else:
        raise ValueError(f"Unknown format: {format}")


def get_timestamp_custom(format_string: str) -> str:
    """
    Get timestamp with custom format string
    
    Args:
        format_string: Python datetime format string
            Common codes:
            %Y - Year (4 digits)
            %m - Month (01-12)
            %d - Day (01-31)
            %H - Hour 24h (00-23)
            %I - Hour 12h (01-12)
            %M - Minute (00-59)
            %S - Second (00-59)
            %p - AM/PM
            %A - Weekday name
            %B - Month name
            
    Returns:
        Formatted timestamp string
        
    Examples:
        >>> get_timestamp_custom("%Y%m%d_%H%M%S")
        '20240115_143045'
        
        >>> get_timestamp_custom("%B %d, %Y at %I:%M %p")
        'January 15, 2024 at 02:30 PM'
    """
    return datetime.now().strftime(format_string)


def timestamp_to_datetime(timestamp_str: str, format: str = "default", separator: str = "-") -> datetime:
    """
    Convert timestamp string back to datetime object
    
    Args:
        timestamp_str: Timestamp string
        format: Format of the timestamp string
        separator: Separator used in timestamp
        
    Returns:
        datetime object
        
    Examples:
        >>> timestamp_to_datetime("2024-01-15-14-30")
        datetime.datetime(2024, 1, 15, 14, 30)
    """
    if format == "default":
        return datetime.strptime(timestamp_str, f"%Y{separator}%m{separator}%d{separator}%H{separator}%M")
    elif format == "full":
        return datetime.strptime(timestamp_str, f"%Y{separator}%m{separator}%d{separator}%H{separator}%M{separator}%S")
    elif format == "compact":
        return datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    elif format == "iso":
        return datetime.fromisoformat(timestamp_str)
    else:
        raise ValueError(f"Unknown format: {format}")


# ============================================================================
# COMBINED UTILITY FUNCTIONS
# ============================================================================

def create_unique_id(prefix: str = "", suffix: str = "", 
                     timestamp_format: str = "default", 
                     hash_length: int = 8,
                     separator: str = "_") -> str:
    """
    Create a unique identifier combining timestamp and hash
    
    Args:
        prefix: Optional prefix for the ID
        suffix: Optional suffix for the ID
        timestamp_format: Format for timestamp portion
        hash_length: Length of random hash portion
        separator: Character to separate components
        
    Returns:
        Unique ID string
        
    Examples:
        >>> create_unique_id()
        '2024-01-15-14-30_a3f7b2c9'
        
        >>> create_unique_id(prefix="img", suffix="jpg")
        'img_2024-01-15-14-30_a3f7b2c9_jpg'
        
        >>> create_unique_id(prefix="video", timestamp_format="compact", hash_length=12)
        'video_20240115143045_k3mp9xqz4rty'
    """
    timestamp = get_timestamp(timestamp_format, separator="-")
    hash_str = generate_hash(hash_length, "lowercase")
    
    components = []
    if prefix:
        components.append(prefix)
    components.append(timestamp)
    components.append(hash_str)
    if suffix:
        components.append(suffix)
    
    return separator.join(components)


def create_filename(base_name: str, extension: str = "", 
                   add_timestamp: bool = True, 
                   add_hash: bool = True,
                   hash_length: int = 8) -> str:
    """
    Create a unique filename with optional timestamp and hash
    
    Args:
        base_name: Base name for the file
        extension: File extension (with or without dot)
        add_timestamp: Whether to add timestamp
        add_hash: Whether to add random hash
        hash_length: Length of hash if used
        
    Returns:
        Filename string
        
    Examples:
        >>> create_filename("image", ".jpg")
        'image_2024-01-15-14-30_a3f7b2c9.jpg'
        
        >>> create_filename("data", "csv", add_hash=False)
        'data_2024-01-15-14-30.csv'
        
        >>> create_filename("report", ".pdf", add_timestamp=False)
        'report_a3f7b2c9.pdf'
    """
    # Ensure extension starts with dot
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    components = [base_name]
    
    if add_timestamp:
        timestamp = get_timestamp("default")
        components.append(timestamp)
    
    if add_hash:
        hash_str = generate_hash(hash_length, "lowercase")
        components.append(hash_str)
    
    filename = "_".join(components) + extension
    return filename


def create_directory_name(base_name: str, add_timestamp: bool = True, 
                         timestamp_format: str = "date") -> str:
    """
    Create a directory name with optional timestamp
    
    Args:
        base_name: Base name for directory
        add_timestamp: Whether to add timestamp
        timestamp_format: Format for timestamp
        
    Returns:
        Directory name string
        
    Examples:
        >>> create_directory_name("output")
        'output_2024-01-15'
        
        >>> create_directory_name("experiment", timestamp_format="full")
        'experiment_2024-01-15-14-30-45'
    """
    if add_timestamp:
        timestamp = get_timestamp(timestamp_format)
        return f"{base_name}_{timestamp}"
    return base_name


# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def generate_random_string(length: int = 10, 
                          include_uppercase: bool = True,
                          include_lowercase: bool = True,
                          include_digits: bool = True,
                          include_special: bool = False) -> str:
    """
    Generate a random string with specified character types
    
    Args:
        length: Length of string
        include_uppercase: Include uppercase letters
        include_lowercase: Include lowercase letters
        include_digits: Include digits
        include_special: Include special characters
        
    Returns:
        Random string
        
    Examples:
        >>> generate_random_string(12)
        'K3mP9xQz4RtY'
        
        >>> generate_random_string(16, include_special=True)
        'aB3$xY9@mN2#pQ5!'
    """
    alphabet = ""
    if include_uppercase:
        alphabet += string.ascii_uppercase
    if include_lowercase:
        alphabet += string.ascii_lowercase
    if include_digits:
        alphabet += string.digits
    if include_special:
        alphabet += string.punctuation
    
    if not alphabet:
        raise ValueError("At least one character type must be included")
    
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def short_uuid(length: int = 8) -> str:
    """
    Generate a short UUID-based identifier
    
    Args:
        length: Desired length (default: 8)
        
    Returns:
        Short UUID string
        
    Examples:
        >>> short_uuid()
        'a3f7b2c9'
        
        >>> short_uuid(12)
        'k3mp9xqz4rty'
    """
    return str(uuid.uuid4()).replace('-', '')[:length]


def ensure_unique_name(base_name: str, existing_names: list, 
                      separator: str = "_", max_attempts: int = 1000) -> str:
    """
    Ensure a name is unique by adding a suffix if needed
    
    Args:
        base_name: Base name to make unique
        existing_names: List of existing names to check against
        separator: Separator before numeric suffix
        max_attempts: Maximum attempts before giving up
        
    Returns:
        Unique name
        
    Examples:
        >>> ensure_unique_name("file", ["file", "file_1", "file_2"])
        'file_3'
        
        >>> ensure_unique_name("test", [])
        'test'
    """
    if base_name not in existing_names:
        return base_name
    
    for i in range(1, max_attempts):
        candidate = f"{base_name}{separator}{i}"
        if candidate not in existing_names:
            return candidate
    
    # If max attempts reached, add random hash
    hash_str = generate_hash(8, "lowercase")
    return f"{base_name}{separator}{hash_str}"


# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def run_examples():
    """Run example usage of utility functions"""
    
    print("="*70)
    print("UTILS.PY EXAMPLES")
    print("="*70)
    
    print("\n1. Hash Generation:")
    print("-" * 50)
    print(f"Hex (8 chars):        {generate_hash(8, 'hex')}")
    print(f"Alphanumeric (12):    {generate_hash(12, 'alphanumeric')}")
    print(f"Lowercase (16):       {generate_hash(16, 'lowercase')}")
    print(f"UUID:                 {generate_hash(method='uuid')}")
    
    print("\n2. Data Hashing:")
    print("-" * 50)
    print(f"MD5 hash:             {generate_md5_hash('hello world')}")
    print(f"MD5 (8 chars):        {generate_md5_hash('hello world', 8)}")
    print(f"SHA256 hash:          {generate_sha256_hash('hello world', 16)}")
    
    print("\n3. Timestamps:")
    print("-" * 50)
    print(f"Default:              {get_timestamp()}")
    print(f"Full:                 {get_timestamp('full')}")
    print(f"Date only:            {get_timestamp('date')}")
    print(f"Compact:              {get_timestamp('compact')}")
    print(f"ISO:                  {get_timestamp('iso')}")
    print(f"Unix:                 {get_timestamp('unix')}")
    
    print("\n4. Unique IDs:")
    print("-" * 50)
    print(f"Basic:                {create_unique_id()}")
    print(f"With prefix:          {create_unique_id(prefix='img')}")
    print(f"With prefix & suffix: {create_unique_id(prefix='video', suffix='mp4')}")
    print(f"Compact:              {create_unique_id(timestamp_format='compact')}")
    
    print("\n5. Filenames:")
    print("-" * 50)
    print(f"Image file:           {create_filename('photo', '.jpg')}")
    print(f"Data file:            {create_filename('data', 'csv')}")
    print(f"No hash:              {create_filename('report', '.pdf', add_hash=False)}")
    print(f"No timestamp:         {create_filename('config', '.json', add_timestamp=False)}")
    
    print("\n6. Directory Names:")
    print("-" * 50)
    print(f"Output dir:           {create_directory_name('output')}")
    print(f"Experiment dir:       {create_directory_name('experiment', timestamp_format='full')}")
    
    print("\n7. Random Strings:")
    print("-" * 50)
    print(f"Alphanumeric:         {generate_random_string(12)}")
    print(f"With special chars:   {generate_random_string(16, include_special=True)}")
    print(f"Numbers only:         {generate_random_string(10, False, False, True, False)}")
    
    print("\n8. Short UUID:")
    print("-" * 50)
    print(f"8 characters:         {short_uuid(8)}")
    print(f"12 characters:        {short_uuid(12)}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    run_examples()