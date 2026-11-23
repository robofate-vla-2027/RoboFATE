"""Verify the SHA-256 inventory of this source release using the standard library."""
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    seen = set()
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        if relative in seen or not path.is_relative_to(root):
            raise ValueError('Invalid inventory path: ' + relative)
        seen.add(relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Checksum mismatch: ' + relative)
    print(f'Verified {len(seen)} source-release files.')


if __name__ == '__main__':
    main()
