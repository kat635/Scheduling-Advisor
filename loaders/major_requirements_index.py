""" Builds an index for major requirements data, 
    including URLs for specific majors, concentrations, and degree types."""

def _normalize_key(key: str) -> str:
    """Normalizes a key by stripping whitespace and converting to lowercase."""
    return key.strip().lower()


def build_index(data):
    """
    Builds an index for major requirements based on the provided data. 
    Normalizes keys and handles aliases for both majors and concentrations. 

    Args: 
        data: A dictionary containing major requirements data, including majors, concentrations, and their aliases.

    Returns:
        A dictionary mapping normalized major and concentration keys to their corresponding URLs for BS and BA degree types
    """
    index = {}

    for major, major_data in data.items():
        major_key = _normalize_key(major)
        default_aliases = [
            _normalize_key(alias)
            for alias in major_data.get("default_aliases", [])
            if alias
        ]

        # Get the first concentration's URLs as default
        default_urls = None
        for conc_data in major_data.get("concentrations", {}).values():
            urls = {}
            if "bs" in conc_data:
                urls["bs"] = conc_data["bs"]
            if "ba" in conc_data:
                urls["ba"] = conc_data["ba"]
            if urls:
                default_urls = urls
                break

        # Index the major key and default aliases
        index[major_key] = default_urls
        for alias in default_aliases:
            index[alias] = default_urls

        for conc, conc_data in major_data.get("concentrations", {}).items():
            conc_key = _normalize_key(conc)
            conc_aliases = [
                _normalize_key(alias)
                for alias in conc_data.get("aliases", [])
                if alias
            ]

            urls = {}
            if "bs" in conc_data:
                urls["bs"] = conc_data["bs"]
            if "ba" in conc_data:
                urls["ba"] = conc_data["ba"]

            # direct major:concentration keys
            index[f"{major_key}:{conc_key}"] = urls if urls else None
            for alias in conc_aliases:
                index[f"{major_key}:{alias}"] = urls if urls else None
            for alias in default_aliases:
                index[f"{alias}:{conc_key}"] = urls if urls else None
            for major_alias in default_aliases:
                for conc_alias in conc_aliases:
                    index[f"{major_alias}:{conc_alias}"] = urls if urls else None

    return index


def get_url(index, major, concentration=None, degree_type=None):
    """
    Retrieves the URL for the major requirements based on the provided major, concentration, and degree type.
    
    Args:
        index: The index built by the build_index function.
        major: The major name (string).
        concentration: The concentration name (string, optional).
        degree_type: The degree type, either 'bs' or 'ba' (string, optional).

    Returns:
        The URL for the major requirements if found, otherwise None.
    """
    # Normalize to lowercase for matching
    major_lower = major.lower() if major else None
    concentration_lower = concentration.lower() if concentration else None
    degree_type_lower = degree_type.lower() if degree_type else None
    
    # Build the key
    if concentration_lower:
        key = f"{major_lower}:{concentration_lower}"
    else:
        key = major_lower
    
    # Get the result (could be a dict with ba/bs urls)
    result = index.get(key)
    
    # If result is a dict with degree_type options, extract the right one
    if isinstance(result, dict):
        if degree_type_lower in result:
            return result[degree_type_lower]
        # If degree_type requested but not found, prefer bs, then ba
        if degree_type_lower:
            return result.get('bs') or result.get('ba')
        # No specific degree requested, prefer bs
        return result.get('bs') or result.get('ba')
    
    return result