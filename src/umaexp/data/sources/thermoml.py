"""IUPAC ThermoML parser (NIST TRC ThermoML Archive).

Files: https://trc.nist.gov/ThermoML/<DOI>.xml (archive covers cooperating
journals through 2019). The parser is namespace-agnostic and returns every
numerical data point with its variables, constraints, phases and
uncertainty definition; `pure_liquid_density` then selects single-component
liquid mass densities.

Uncertainties: ThermoML stores combined *expanded* uncertainties with a level
of confidence (usually 95 %). Standard sd = U / k with k = 2 for 95 %
(recorded as the conversion rule; k=1.96 would change sd by 2 %).
Mixture data, solutions and derived quantities are left out of the pilot
density selection but are parsed.
"""

from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET

from ..schema import (
    Conditions,
    Observation,
    Phase,
    Provenance,
    Redistribution,
    SplitGroups,
    TransformStep,
    Uncertainty,
    UncertaintyType,
    ValueKind,
)

BASE = "https://trc.nist.gov/ThermoML/"
PARSER_VERSION = "thermoml_v1"


def _s(tag):
    return tag.split("}")[-1]


def _find(el, name):
    for c in el.iter():
        if _s(c.tag) == name:
            return c
    return None


def _text(el, name):
    c = _find(el, name)
    return None if c is None or c.text is None else c.text.strip()


def parse(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    compounds = {}
    for c in root:
        if _s(c.tag) != "Compound":
            continue
        org = _text(c, "nOrgNum")
        names = [n.text for n in c.iter() if _s(n.tag) == "sCommonName"]
        compounds[org] = {"inchi": _text(c, "sStandardInChI"), "inchikey": _text(c, "sStandardInChIKey"),
                          "names": names, "formula": _text(c, "sFormulaMolec")}
    citation = {"doi": _text(root, "sDOI"), "title": _text(root, "sTitle"), "journal": _text(root, "sPubName")}
    datasets = []
    for d in root:
        if _s(d.tag) != "PureOrMixtureData":
            continue
        comps = [_text(c, "nOrgNum") for c in d if _s(c.tag) == "Component"]
        props = {}
        for p in d:
            if _s(p.tag) != "Property":
                continue
            num = _text(p, "nPropNumber")
            name = next((x.text for x in p.iter() if _s(x.tag) == "ePropName"), None)
            unc = _find(p, "CombinedUncertainty")
            props[num] = {"name": name, "phase": _text(p, "ePropPhase"),
                          "method": next((x.text for x in p.iter() if _s(x.tag) == "eMethodName"), None),
                          "conf": float(_text(unc, "nCombUncertLevOfConfid")) if unc is not None and _text(unc, "nCombUncertLevOfConfid") else None}
        constraints = []
        for c in d:
            if _s(c.tag) != "Constraint":
                continue
            ctype = next((x for x in _find(c, "ConstraintType")), None) if _find(c, "ConstraintType") is not None else None
            constraints.append({"type": None if ctype is None else ctype.text, "value": float(_text(c, "nConstraintValue"))})
        variables = {}
        for v in d:
            if _s(v.tag) != "Variable":
                continue
            vtype = next((x for x in _find(v, "VariableType")), None)
            variables[_text(v, "nVarNumber")] = None if vtype is None else vtype.text
        points = []
        for nv in d:
            if _s(nv.tag) != "NumValues":
                continue
            vars_ = {variables.get(_text(vv, "nVarNumber")): float(_text(vv, "nVarValue"))
                     for vv in nv if _s(vv.tag) == "VariableValue"}
            for pv in nv:
                if _s(pv.tag) != "PropertyValue":
                    continue
                pnum = _text(pv, "nPropNumber")
                u = _text(pv, "nCombExpandUncertValue")
                points.append({"property": props.get(pnum, {}), "value": float(_text(pv, "nPropValue")),
                               "expanded_uncertainty": float(u) if u else None, "variables": vars_})
        datasets.append({"components": comps, "constraints": constraints, "points": points,
                         "phase": _text(d, "ePhase")})
    return {"compounds": compounds, "datasets": datasets, "citation": citation}


def pure_liquid_density(parsed: dict, doi: str, fetch_info: dict, system_id_for_inchikey) -> list[Observation]:
    """Single-component liquid mass densities as Observations (g/cm^3)."""
    out = []
    for di, d in enumerate(parsed["datasets"]):
        if len(d["components"]) != 1:
            continue
        comp = parsed["compounds"].get(d["components"][0], {})
        ik = comp.get("inchikey")
        sid = system_id_for_inchikey(ik) if ik else None
        if sid is None:
            continue
        p_const = next((c["value"] for c in d["constraints"] if c["type"] and c["type"].startswith("Pressure, kPa")), None)
        for k, pt in enumerate(d["points"]):
            prop = pt["property"]
            if not (prop.get("name") or "").startswith("Mass density, kg/m3") or (prop.get("phase") or "") != "Liquid":
                continue
            T = pt["variables"].get("Temperature, K")
            p_kpa = pt["variables"].get("Pressure, kPa", p_const)
            U = pt["expanded_uncertainty"]
            conf = prop.get("conf")
            missing = [n for n, v in (("temperature", T), ("pressure", p_kpa)) if v is None]
            unc = (Uncertainty(kind=UncertaintyType.EXPANDED, value=U / 1000.0, coverage_factor=2.0,
                               source_text=f"combined expanded uncertainty, level of confidence {conf}%",
                               conversion_rule="k=2 for 95% confidence")
                   if U is not None else Uncertainty(kind=UncertaintyType.UNKNOWN))
            out.append(Observation(
                observation_id=f"thermoml:{doi}:{di}:{k}",
                system_ids=[sid],
                family="density",
                quantity="mass_density",
                definition="Pure-liquid mass density at (T, p)",
                value=pt["value"] / 1000.0,
                units="g/cm^3",
                labels={"method": prop.get("method"), "compound": comp.get("names", [None])[0], "inchikey": ik},
                uncertainty=unc,
                value_kind=ValueKind.MEASURED,
                conditions=Conditions(temperature_k=T, pressure_bar=None if p_kpa is None else p_kpa / 100.0,
                                      phase=Phase.LIQUID, protocol=prop.get("method"), missing=missing),
                compatible_uma_tasks=["omol (periodic, acknowledged)", "omc (crystal-trained)"],
                forward_model={"name": "npt_density", "estimator": "mass_over_mean_volume"},
                provenance=Provenance(
                    source_name="NIST TRC ThermoML Archive", source_url=fetch_info["url"], doi=doi,
                    source_record_id=f"{doi}#dataset{di}/point{k}",
                    retrieval_date=dt.date.fromisoformat(fetch_info["retrieval_date"]),
                    license="ThermoML Archive terms (see trc.nist.gov/ThermoML); article data",
                    redistribution=Redistribution.UNKNOWN,
                    citation=parsed["citation"].get("title"),
                    transformation_history=[TransformStep(step="parse_thermoml", tool="umaexp",
                                                          version=PARSER_VERSION,
                                                          parameters={"sha256": fetch_info["sha256"]})],
                ),
                groups=SplitGroups(molecule_group=ik.split("-")[0], provenance_group="thermoml",
                                   publication_group=doi),
                quality_flags=[],
            ))
    return out
