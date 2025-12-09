"""Partner metadata shared across the UI."""

from __future__ import annotations

from typing import Dict, List

PARTNER_ORGANIZATIONS: List[Dict[str, str]] = [
    {
        "name": "Uppsala University",
        "logo": "images/uu_logo.png",
        "url": "https://www.uu.se/en/",
        "size": "xl",
    },
    {
        "name": "REMEDi4ALL",
        "logo": "images/remedi4all_logo.png",
        "url": "https://www.remedi4all.org/",
        "size": "xl"
    },
    {
        "name": "Fraunhofer ITMP",
        "logo": "images/ITMP_logo.png",
        "url": "https://www.itmp.fraunhofer.de/en.html",
        "size": "xl",
    },
    {
        "name": "Karolinska Institutet",
        "logo": "images/KI_logo.png",
        "url": "https://ki.se/en",
        "size": "xl",
    },
    {
        "name": "FIMM",
        "logo": "images/FIMM_logo.png",
        "url": "https://www.helsinki.fi/en/hilife-helsinki-institute-life-science/units/fimm",
        "size": "xl",
    },
    {
        "name": "EATRIS",
        "logo": "images/eatris_logo.png",
        "url": "https://eatris.eu/",
        "size": "xl"
    },
    {
        "name": "SciLifeLab Serve",
        "logo": "images/serve_logo.png",
        "url": "https://serve.scilifelab.se/",
        "size": "xl"
    },
    {
        "name": "Chemical Biology Consortium Sweden",
        "logo": "images/CBCS_logo.png",
        "url": "https://www.cbcs.se",
        "size": "xl"
    },
    {
        "name": "European Union",
        "logo": "images/EU_logo.png",
        "url": "https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en",
        "size": "xl"
    }
]


def get_partner_organizations() -> List[Dict[str, str]]:
    """Return a shallow copy of partner metadata."""
    return [dict(partner) for partner in PARTNER_ORGANIZATIONS]
