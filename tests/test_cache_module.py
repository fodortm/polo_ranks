from infra.cache import build_file_fingerprint

def test_fingerprint_empty_list():
    assert isinstance(build_file_fingerprint([]), str)
