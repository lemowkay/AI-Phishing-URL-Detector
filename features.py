
import ipaddress
from urllib.parse import urlsplit


def extract_url_features(url):
    """Extract numerical features from a URL without visiting it."""

    url = str(url).strip()

   
    url_to_parse = url
    if "://" not in url:
        url_to_parse = "http://" + url

    try:
        parsed = urlsplit(url_to_parse)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        query = parsed.query.lower()
        scheme = parsed.scheme.lower()

    except ValueError:
        hostname = ""
        path = ""
        query = ""
        scheme = ""

    # Check whether the hostname is an IP address
    is_ip = 0

    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        pass

   
    suspicious_words = [
        "login",
        "verify",
        "secure",
        "account",
        "update",
        "password",
        "signin",
        "confirm"
    ]

    suspicious_count = sum(
        word in url.lower()
        for word in suspicious_words
    )

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(char.isdigit() for char in url),
        "num_at_symbols": url.count("@"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "uses_https": int(scheme == "https"),
        "is_ip_address": is_ip,
        "suspicious_word_count": suspicious_count
    }
