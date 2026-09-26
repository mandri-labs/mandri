import base64
import datetime
import ipaddress
import re
import secrets
import string
import uuid

EMAIL = re.compile(
    r"(?<![\w.!#$%&'*+/=?^`{|}~-])[\w.!#$%&'*+/=?^`{|}~-]{1,64}"
    r"@(?:[^\W_][\w-]{0,62}\.){1,20}[^\W_][\w-]{0,62}",
    re.UNICODE,
)
MAC = re.compile(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}|(?:[0-9a-fA-F]{2}-){5}[0-9a-fA-F]{2}")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
IBAN_LENGTHS = {
    "AT": 20,
    "BE": 16,
    "BG": 22,
    "CH": 21,
    "CY": 28,
    "CZ": 24,
    "DE": 22,
    "DK": 18,
    "EE": 20,
    "ES": 24,
    "FI": 18,
    "FR": 27,
    "GB": 22,
    "GR": 27,
    "HR": 21,
    "HU": 28,
    "IE": 22,
    "IS": 26,
    "IT": 27,
    "LI": 21,
    "LT": 20,
    "LU": 20,
    "LV": 21,
    "MC": 27,
    "MT": 31,
    "NL": 18,
    "NO": 15,
    "PL": 28,
    "PT": 25,
    "RO": 24,
    "SE": 24,
    "SI": 19,
    "SK": 24,
    "SM": 27,
}
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y", "%m/%d/%Y")


def shape(value: str) -> str:
    result = []
    for char in value:
        if char in string.ascii_lowercase:
            result.append(secrets.choice(string.ascii_lowercase))
        elif char in string.ascii_uppercase:
            result.append(secrets.choice(string.ascii_uppercase))
        elif char in string.digits:
            result.append(secrets.choice(string.digits))
        elif char.isalpha():
            alphabet = string.ascii_uppercase if char.isupper() else string.ascii_lowercase
            result.append(secrets.choice(alphabet))
        else:
            result.append(char)
    return "".join(result)


def luhn(value: str) -> bool:
    digits = re.sub(r"[ -]", "", value)
    if not digits.isascii() or not digits.isdigit() or not 12 <= len(digits) <= 19:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        number = int(digit)
        if index % 2:
            number *= 2
            number = number - 9 if number > 9 else number
        checksum += number
    return checksum % 10 == 0


def iban_valid(value: str) -> bool:
    compact = value.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    if IBAN_LENGTHS.get(compact[:2]) != len(compact):
        return False
    rotated = compact[4:] + compact[:4]
    digits = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rotated)
    return int(digits) % 97 == 1


def regroup(template: str, compact: str) -> str:
    iterator = iter(compact)
    return "".join(char if char in " -" else next(iterator) for char in template)


def date_format(value: str) -> str | None:
    for form in DATE_FORMATS:
        try:
            date = datetime.datetime.strptime(value, form).date()
        except ValueError:
            continue
        if date.strftime(form) == value:
            return form
    return None


def phone_valid(value: str) -> bool:
    if not re.fullmatch(r"\+?[0-9() .-]{7,30}", value):
        return False
    digits = "".join(char for char in value if char.isdigit())
    if not 7 <= len(digits) <= 15:
        return False
    if value.startswith("+33"):
        return len(digits) == 11 and digits[2] in "123456789"
    if value.startswith("+1"):
        return len(digits) == 11 and digits[1] in "23456789" and digits[4] in "23456789"
    if value.startswith("0"):
        return len(digits) == 10 and digits[1] in "123456789"
    return not value.startswith("+") or digits[0] != "0"


def valid(kind: str, value: str, original: str | None = None) -> bool:
    try:
        if not value:
            return False
        if kind == "email":
            if EMAIL.fullmatch(value) is None:
                return False
            local, domain = value.rsplit("@", 1)
            domain.encode("idna")
            return len(local) <= 64 and len(domain) <= 253
        if kind in {"ipv4", "ipv6"}:
            address = ipaddress.ip_address(value.split("%", 1)[0])
            return address.version == (4 if kind == "ipv4" else 6)
        if kind == "cidr":
            network = ipaddress.ip_network(value, strict=False)
            if original is None:
                return True
            previous = ipaddress.ip_network(original, strict=False)
            return network.version == previous.version and network.prefixlen == previous.prefixlen
        if kind == "mac":
            return MAC.fullmatch(value) is not None
        if kind == "uuid":
            return UUID.fullmatch(value) is not None
        if kind == "iban":
            return iban_valid(value)
        if kind == "card":
            return luhn(value)
        if kind == "date_of_birth":
            return date_format(value) is not None
        if kind == "phone":
            return phone_valid(value)
        if kind == "basic":
            return ":" in base64.b64decode(value, validate=True).decode("utf-8")
        if kind in {"domain", "hostname"}:
            return all(
                re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
                for label in value.rstrip(".").split(".")
            )
        if kind in {"git_owner", "git_repository"}:
            return re.fullmatch(r"[a-zA-Z0-9_.-]+", value) is not None
        if kind == "private_package":
            return re.fullmatch(r"@[\w.-]+/[\w.-]+", value) is not None
        if kind == "cloud_resource":
            return value.startswith("arn:") and len(value.split(":", 5)) == 6
        if kind in {"identifier", "account", "secret"} and original:
            if re.fullmatch(r"-?(?:0|[1-9]\d*)", original) and not re.fullmatch(
                r"-?(?:0|[1-9]\d*)", value
            ):
                return False
            return len(value) == len(original) and all(
                (left.isdigit() and right.isdigit())
                or (left.isalpha() and right.isalpha() and left.isupper() == right.isupper())
                or left == right
                for left, right in zip(original, value, strict=True)
            )
        return True
    except (ValueError, UnicodeError):
        return False


class SurrogateGenerator:
    def generate(self, kind: str, original: str) -> str:
        if not valid(kind, original):
            return original
        if kind == "email":
            local = secrets.token_hex(12)
            return local + "@" + secrets.token_hex(8) + ".invalid"
        if kind == "ipv4":
            return str(
                ipaddress.IPv4Address(
                    int(ipaddress.IPv4Address("198.18.0.0")) + secrets.randbelow(131070) + 1
                )
            )
        if kind == "ipv6":
            number = int(ipaddress.IPv6Address("2001:db8::")) + secrets.randbits(96)
            result = str(ipaddress.IPv6Address(number))
            if "%" in original:
                result += "%" + shape(original.split("%", 1)[1])
            return result
        if kind == "cidr":
            network = ipaddress.ip_network(original, strict=False)
            address = self.generate(
                "ipv4" if network.version == 4 else "ipv6", str(network.network_address)
            )
            return str(ipaddress.ip_network(f"{address}/{network.prefixlen}", strict=False))
        if kind == "mac":
            parts = [f"{secrets.randbelow(256):02x}" for _ in range(6)]
            parts[0] = f"{int(parts[0], 16) & 0xFC | 0x02:02x}"
            result = (":" if ":" in original else "-").join(parts)
            return result.upper() if original.upper() == original else result
        if kind == "uuid":
            original_uuid = uuid.UUID(original)
            randomized = bytearray(secrets.token_bytes(16))
            randomized[6] = (randomized[6] & 0x0F) | (original_uuid.bytes[6] & 0xF0)
            randomized[8] = (randomized[8] & 0x3F) | (original_uuid.bytes[8] & 0xC0)
            result = str(uuid.UUID(bytes=bytes(randomized)))
            return result.upper() if original.upper() == original else result
        if kind == "iban":
            compact = original.replace(" ", "").upper()
            bban = shape(compact[4:])
            rearranged = bban + compact[:2] + "00"
            iban_number = "".join(
                str(ord(char) - 55) if char.isalpha() else char for char in rearranged
            )
            result = compact[:2] + f"{98 - int(iban_number) % 97:02d}" + bban
            return regroup(original, result.lower() if original.islower() else result)
        if kind == "card":
            length = sum(char.isdigit() for char in original)
            prefix = "4" + "".join(secrets.choice(string.digits) for _ in range(length - 2))
            result = next(prefix + str(digit) for digit in range(10) if luhn(prefix + str(digit)))
            return regroup(original, result)
        if kind == "date_of_birth":
            form = date_format(original)
            if form is None:
                return original
            date = datetime.date(1940, 1, 1) + datetime.timedelta(days=secrets.randbelow(24000))
            return date.strftime(form)
        if kind == "phone":
            chars = list(shape(original))
            positions = [index for index, char in enumerate(original) if char.isdigit()]
            preserve: tuple[int, ...]
            if original.startswith("0"):
                preserve = (0, 1)
            elif original.startswith("+33"):
                preserve = (0, 1, 2)
            elif original.startswith("+1"):
                preserve = (0, 1, 4)
            else:
                preserve = (0, 1) if original.startswith("+") else ()
            for position in preserve:
                chars[positions[position]] = original[positions[position]]
            return "".join(chars)
        if kind == "basic":
            try:
                decoded = base64.b64decode(original, validate=True).decode("utf-8")
            except (ValueError, UnicodeError):
                return original
            if ":" not in decoded:
                return original
            return base64.b64encode(shape(decoded).encode()).decode()
        if kind in {"domain", "hostname"}:
            labels = original.rstrip(".").split(".")
            result = (
                ".".join(shape(label) for label in labels[:-1]) + ".invalid"
                if len(labels) > 1
                else shape(labels[0])
            )
            return result + ("." if original.endswith(".") else "")
        if kind == "secret":
            secret_prefix = re.match(
                r"(?:sk-(?:proj-|ant-api\d+-)?|gh[pousr]_|github_pat_|AKIA|ASIA)", original
            )
            return (
                (secret_prefix.group() + shape(original[secret_prefix.end() :]))
                if secret_prefix
                else shape(original)
            )
        result = shape(original)
        if re.fullmatch(r"-?[1-9]\d*", original):
            start = int(result.startswith("-"))
            result = result[:start] + secrets.choice("123456789") + result[start + 1 :]
        return result
