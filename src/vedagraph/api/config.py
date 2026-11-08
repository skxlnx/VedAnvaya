"""Environment-driven API configuration.

Deliberately a separate settings object from :class:`vedagraph.config.Settings`. That one
carries the ``VEDAGRAPH_`` prefix and describes the ingestion side -- rate limits, user
agents, data directories -- none of which the HTTP layer has any business reading. The
API's variables are unprefixed (``NEO4J_URI``, ``API_PORT``) because that is what a
deployment target expects to set, and mixing the two namespaces would mean an operator
editing a scraper's throttle to change a database password.

Nothing here is ever returned to a client. :func:`ApiSettings.safe_summary` exists so that
``/ready`` can describe the connection without disclosing it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Bound on every paginated collection. A client asking for more gets 422 rather than a
#: silently truncated page, because silent truncation is indistinguishable from "that is
#: all there is" and this API's whole discipline is that absence must be explicit.
MAX_PAGE_SIZE: Final = 200

#: Default page size where the client says nothing.
DEFAULT_PAGE_SIZE: Final = 25

#: Bound on graph fan-out. The neighbourhood endpoint multiplies this by depth, so it is
#: much smaller than MAX_PAGE_SIZE.
MAX_NEIGHBOURS_PER_TYPE: Final = 50

#: The ontology version this API is written against. ``/ready`` fails if the graph does
#: not carry it: a frontend talking to a graph built under a different model would get
#: silently wrong shapes rather than an error.
EXPECTED_DOMAIN_MODEL_VERSION: Final = "vedagraph-knowledge-model-v2"

#: The four works this API expects to exist. ``/ready`` checks them by id rather than by
#: count, so a corpus swapped for another of the same size is still caught.
EXPECTED_WORK_IDS: Final[tuple[str, ...]] = (
    "VG:WORK:RV:SAK",
    "VG:WORK:SV:KAU",
    "VG:WORK:YV:VSM",
    "VG:WORK:AV:SAU",
)


class ApiSettings(BaseSettings):
    """Connection and server settings, read from the environment or ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("neo4j")
    neo4j_database: str = "neo4j"

    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_debug: bool = False

    #: ``aura_free`` is a deliberately reduced, public deployment profile.  It omits the
    #: internal Rigvedic lemma edge index so the graph stays inside AuraDB Free's relationship
    #: ceiling.  Local and full deployments use the default and retain every graph edge.
    api_deployment_profile: Literal["full", "aura_free"] = "full"

    #: Root of the committed data tree. The API reads exactly two things from it -- the
    #: audio catalog and its optional media cache -- and never the corpus, which reaches
    #: this process only through Neo4j. Named without the ``VEDAGRAPH_`` prefix like the
    #: rest of this object because it is a deployment path, not an ingestion setting.
    data_dir: Path = Path("data")

    #: Limits on POST /api/v1/ask, which is billed per call. Named with the ``VEDAGRAPH_``
    #: prefix, unlike the rest of this object, because they belong with the LLM settings an
    #: operator already sets under that prefix. A daily cap of 0 refuses every question.
    ask_rate_limit_per_minute: int = Field(
        default=10, ge=1, validation_alias="VEDAGRAPH_ASK_RATE_LIMIT_PER_MINUTE"
    )
    ask_daily_cap: int = Field(default=1000, ge=0, validation_alias="VEDAGRAPH_ASK_DAILY_CAP")
    #: How many proxies sit in front of this process and append to ``X-Forwarded-For``. Sets
    #: which entry is read as the client address; see ``ratelimit.client_ip``.
    trusted_proxies: int = Field(default=1, ge=1, validation_alias="VEDAGRAPH_TRUSTED_PROXIES")

    #: Seconds before a Cypher call is abandoned. A product read that takes longer than
    #: this is a defect, not a slow query worth waiting for.
    neo4j_query_timeout_seconds: float = Field(default=15.0, gt=0, le=120)

    @property
    def auth(self) -> tuple[str, str]:
        return (self.neo4j_user, self.neo4j_password.get_secret_value())

    def safe_summary(self) -> dict[str, Any]:
        """Connection description with the password and user removed.

        ``/ready`` needs to say *which* graph it checked, and an operator debugging a
        misconfigured deployment needs the URI. Neither needs the credentials, so they
        cannot be in the object that reaches the response model at all.
        """
        return {"uri": self.neo4j_uri, "database": self.neo4j_database}


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    """Process-wide settings. Cached because reading ``.env`` per request is silly."""
    return ApiSettings()
