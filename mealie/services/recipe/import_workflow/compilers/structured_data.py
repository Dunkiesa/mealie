import json
from typing import Any

import bs4

from mealie.schema.openai.compiled_source import OpenAICompiledSource
from mealie.services.openai.content import extract_json_ld_data_from_html, find_image
from mealie.services.scraper import cleaner

from ..context import WorkflowContext
from .base import SourceCompiler, SourceType


def _looks_like_json(content: str) -> bool:
    stripped = content.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_like_html(content: str) -> bool:
    return "<" in content and ">" in content


class StructuredDataCompiler(SourceCompiler):
    """
    Compiles sources that are already usable as-is: a schema.org JSON object, HTML containing
    ld+json data, or plain text. These need no AI provider, which keeps single-call imports
    at a single call.
    """

    source_type = SourceType.CONTENT

    def __init__(self, ctx: WorkflowContext, content: str | None = None) -> None:
        super().__init__(ctx, content)
        self._soup: bs4.BeautifulSoup | None = None

    @property
    def soup(self) -> bs4.BeautifulSoup:
        if self._soup is None:
            self._soup = bs4.BeautifulSoup(self.content or "", "lxml")

        return self._soup

    def can_compile(self) -> bool:
        if not self.content:
            return False

        if _looks_like_json(self.content):
            return True

        if _looks_like_html(self.content):
            # HTML is only usable directly if the page carries structured data
            return bool(extract_json_ld_data_from_html(self.soup))

        # plain text is already a faithful record of itself
        return True

    def _image_from_json(self, content: str) -> str | None:
        try:
            data: Any = json.loads(content)
        except json.JSONDecodeError:
            return None

        if isinstance(data, list):
            data = next((item for item in data if isinstance(item, dict)), None)
        if not isinstance(data, dict):
            return None

        return cleaner.clean_image(data.get("image"), default="")[0] or None

    def _url_from_json(self, content: str) -> str | None:
        try:
            data: Any = json.loads(content)
        except json.JSONDecodeError:
            return None

        if isinstance(data, list):
            data = next((item for item in data if isinstance(item, dict)), None)
        if not isinstance(data, dict):
            return None

        for key in ("url", "@id"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        main_entity = data.get("mainEntityOfPage")
        if isinstance(main_entity, str) and main_entity.strip():
            return main_entity.strip()
        if isinstance(main_entity, dict):
            for key in ("url", "@id", "@type"):
                value = main_entity.get(key)
                if isinstance(value, str) and value.strip() and key != "@type":
                    return value.strip()

        same_as = data.get("sameAs")
        if isinstance(same_as, str) and same_as.strip():
            return same_as.strip()
        if isinstance(same_as, list):
            for item in same_as:
                if isinstance(item, str) and item.strip():
                    return item.strip()

        return None

    async def compile(self) -> OpenAICompiledSource | None:
        content = self.content or ""
        image_url: str | None = None
        org_url: str | None = None

        if _looks_like_json(content):
            image_url = self._image_from_json(content)
            org_url = self._url_from_json(content)
        elif _looks_like_html(content):
            image_url = find_image(self.soup)
            content = self.soup.get_text(separator="\n", strip=True) + extract_json_ld_data_from_html(self.soup)
            # the page URL itself is the source URL; compilers keep ctx.input.url separately

        content = content.strip()
        if not content:
            return None

        return OpenAICompiledSource(
            contains_recipe=True,
            content=content,
            language=None,
            image_url=image_url,
            org_url=org_url,
        )
