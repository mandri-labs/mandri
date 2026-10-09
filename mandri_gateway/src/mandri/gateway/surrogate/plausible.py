import re
import secrets
import unicodedata

from faker import Faker


class PlausibleValues:
    def __init__(self) -> None:
        self.faker = Faker("en_US")
        self.faker.seed_instance(secrets.randbits(256))

    def word(self, value: str) -> str:
        ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", "", ascii_value.lower()) or "cedar"

    def domain(self, original: str) -> str:
        count = len(original.rstrip(".").split("."))
        labels = [self.word(self.faker.domain_word())[:40] for _ in range(max(1, count - 1))]
        return ".".join([*labels, "com"]) + ("." if original.endswith(".") else "")

    def email(self) -> str:
        first = self.word(self.faker.first_name())[:28]
        last = self.word(self.faker.last_name())[:28]
        return first + "." + last + "@" + self.domain("example.com")

    def identity(self, context: str) -> str:
        normalized = context.casefold().replace("-", "_")
        if normalized in {"company", "organization", "organisation", "team"}:
            return self.faker.company()
        if normalized == "first_name":
            return self.faker.first_name()
        if normalized == "last_name":
            return self.faker.last_name()
        return self.faker.first_name() + " " + self.faker.last_name()

    def address(self) -> str:
        return self.faker.street_address()
