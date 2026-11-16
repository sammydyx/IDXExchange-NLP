"""Week 8: offline sentence extraction and source-grounded listing summaries."""
from __future__ import annotations

from collections.abc import Mapping
import html
import math
import re
from typing import Any

from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

from scripts.signal_extraction import SignalExtractor


class ListingSummarizer:
    """Select original sentences and retain MLS fields as separate metadata.

    No model downloads are needed. The primary prose is copied from remarks.
    MLS numeric facts remain separate metadata; the old template is comparison
    output only. Every positive feature retains a source sentence.

    splits listing remarks into sentences
    scores them based on bedroom and bathroom counts, price, location, property features,
    and important repair information.
    It selects up to three sentences, reduces repeated information,
    and keeps the original sentence order.
    """

    CATEGORY_WEIGHTS = {"amenity": 5, "exterior_feature": 4, "interior_feature": 3,
                        "layout": 2, "condition": 2, "location_feature": 2}
    GENERIC_TERMS = {"kitchen", "bedroom", "bathroom", "living space", "financing"}
    GENERIC_TERMS |= {"community amenities", "resort-style amenities", "main level", "roof"}
    FEATURE_PRIORITY = {"pool": 12, "spa": 11, "hot tub": 11, "tennis courts": 10,
                        "ocean views": 10, "mountain views": 10, "city views": 9,
                        "clubhouse": 9, "fireplace": 9, "attached garage": 9,
                        "detached garage": 9, "two-car garage": 9, "solar panels": 10,
                        "outdoor kitchen": 9, "rooftop deck": 10, "cul-de-sac": 9,
                        "walk-in closet": 8, "covered patio": 8, "private patio": 8}

    def __init__(self, taxonomy_path=None):
        self.signals = SignalExtractor(taxonomy_path)
        parameters = PunktParameters()
        parameters.abbrev_types = {"sq", "ft", "approx", "st", "ave", "dr", "mr", "mrs", "ca"}
        self.tokenizer = PunktSentenceTokenizer(parameters)

    @staticmethod
    def text(value: Any) -> str:
        if value is None:
            return ""
        value = str(value).strip()
        return "" if value.lower() in {"nan", "none", "<na>", "nat"} else value

    def split_sentences(self, remarks: Any) -> list[str]:
        text = html.unescape(self.text(remarks))
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Some MLS descriptions omit the space after a sentence-ending period.
        text = re.sub(r"(?<=[a-z])\.(?=[A-Z][a-z])", ". ", text)
        return self.tokenizer.tokenize(text) if text else []

    @staticmethod
    def _positive_context(sentence: str, start: int, end: int, term: str) -> bool:
        before, after = sentence[max(0, start - 65):start], sentence[end:end + 65]
        negation = r"\b(?:no|without|not|lacks?|excluding)\b(?:\s+\w+){0,4}[\s-]*$"
        if re.search(negation, before) and not re.search(r"\bnot only\b", before):
            return False
        if re.search(r"\b(?:room|space|potential|plans?)\s+(?:for|to add)(?:\s+a)?\s*$", before):
            return False
        if re.search(r"\b(?:could|can|may)\s+(?:add|build|install)(?:\s+a)?\s*$", before):
            return False
        if re.search(r"\b(?:can serve as|could serve as|optional|potential)\b", before[-45:]):
            return False
        if term == "spa" and re.match(r"[\s-]*(?:inspired|like|style)\b", after):
            return False
        if "garage" in term and (re.search(r"\bconverted\s*$", before) or
                                 re.search(r"\b(?:converted|conversion)\b", after[:50])):
            return False
        if re.search(r"\b(?:near|nearby|close to)\s+(?:a |the )?$", before) and term in {"pool", "spa", "gym"}:
            return False
        return True

    def feature_evidence(self, remarks: Any) -> list[dict[str, Any]]:
        candidates = []
        for index, sentence in enumerate(self.split_sentences(remarks)):
            normalized = self.signals._normalize(sentence)
            for item in self.signals.taxonomy_terms:
                weight = self.CATEGORY_WEIGHTS.get(item["category"], 0)
                if not weight or item["term"] in self.GENERIC_TERMS:
                    continue
                for match in item["pattern"].finditer(normalized):
                    if self._positive_context(normalized, match.start(), match.end(), item["term"]):
                        candidates.append({"term": item["term"], "category": item["category"],
                                           "sentence": sentence, "sentence_index": index,
                                           "score": self.FEATURE_PRIORITY.get(item["term"], weight) + min(len(item["term"].split()), 3) * .2})
                        break
        ranked = sorted(candidates, key=lambda item: (-item["score"], item["sentence_index"], item["term"]))
        unique = []
        for item in ranked:
            term = self.signals._normalize(item["term"])
            if any(term == self.signals._normalize(other["term"]) or
                   re.search(r"\b" + re.escape(term) + r"\b", self.signals._normalize(other["term"]))
                   for other in unique):
                continue
            unique.append(item)
        return unique

    def baseline_extractive_summary(self, remarks: Any, entities: Mapping | None = None,
                                    num_sentences: int = 2) -> str:
        """Previous sentence scorer, retained for same-reference comparisons."""
        if isinstance(num_sentences, bool) or not isinstance(num_sentences, int) or not 1 <= num_sentences <= 3:
            raise ValueError("num_sentences must be 1, 2, or 3")
        sentences = self.split_sentences(remarks)
        evidence = self.feature_evidence(remarks)
        scores = []
        for i, sentence in enumerate(sentences):
            score = 1.5 if i == 0 else 0
            score += sum(min(item["score"], 5) for item in evidence if item["sentence_index"] == i)
            score += 2 * len(set(re.findall(r"\b(?:bedrooms?|bathrooms?|beds?|baths?)\b", sentence.lower())))
            score += 2 * bool(re.search(r"\$\s*\d", sentence))
            for name in ("bedrooms", "bathrooms", "city"):
                value = self.text((entities or {}).get(name))
                if value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", sentence, re.I):
                    score += 1
            if re.search(r"\b(?:call today|schedule|do not disturb|buyer to verify|go direct)\b", sentence, re.I):
                score -= 4
            # A long marketing sentence has a cost, while its facts remain intact.
            score -= max(0, len(sentence.split()) - 45) * .08
            scores.append((score / max(1, len(sentence.split())) ** .5, i))
        selected, seen = [], []
        for _, i in sorted(scores, key=lambda row: (-row[0], row[1])):
            words = set(re.findall(r"\w+", sentences[i].lower()))
            if any(len(words & other) / max(1, len(words | other)) > .85 for other in seen):
                continue
            selected.append(i)
            seen.append(words)
            if len(selected) == num_sentences:
                break
        return " ".join(sentences[i] for i in sorted(selected))

    @staticmethod
    def _room_counts(sentence: str, kind: str) -> list[float]:
        """Count evidence near the noun, including words and adjectives."""
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "single": 1}
        number = r"(?:\d+(?:\.\d+)?(?:\s+\d+\s*/\s*\d+)?|" + "|".join(words) + ")"
        noun = r"(?:bedrooms?|beds?|bdrms?|br|bd)" if kind == "beds" else r"(?:bathrooms?|baths?|ba)"
        pattern = r"(?<![\w/])(" + number + r")([\s-]+(?:[a-z]+[\s-]+){0,3}|\s*)" + noun + r"\b"
        values = []
        for match in re.finditer(pattern, sentence.lower()):
            if re.search(r"\b(?:garage|cars?|bathrooms?|bedrooms?)\b", match.group(2)):
                continue
            raw = match.group(1)
            if raw in words:
                values.append(float(words[raw]))
            elif " " in raw and "/" in raw:
                whole, part = raw.split(None, 1)
                numerator, denominator = part.split("/")
                if float(denominator):
                    values.append(float(whole) + float(numerator) / float(denominator))
            else:
                values.append(float(raw))
        return values

    def extractive_details(self, remarks: Any, entities: Mapping | None = None,
                           num_sentences: int = 3) -> dict[str, Any]:
        """Greedily cover distinct facts while retaining complete original sentences.

        Entity values affect scores only when present in source text. They never
        become generated prose. Feature-list counts are capped, so an elaborate
        kitchen description cannot swamp explicit whole-property room counts.
        """
        if isinstance(num_sentences, bool) or not isinstance(num_sentences, int) or not 1 <= num_sentences <= 3:
            raise ValueError("num_sentences must be 1, 2, or 3")
        entities = entities or {}
        sentences = self.split_sentences(remarks)
        features = self.feature_evidence(remarks)
        feature_weights = {"feature:" + item["term"]: min(3., max(.75, item["score"] / 4))
                           for item in features}
        candidates = []
        for index, sentence in enumerate(sentences):
            lower = sentence.lower()
            rooms = {key: self._room_counts(sentence, key) for key in ("beds", "baths")}
            roles = {key for key, values in rooms.items() if values}
            city = self.text(entities.get("city"))
            if city and re.search(r"(?<!\w)" + re.escape(city) + r"(?!\w)", sentence, re.I):
                roles.add("location")
            if re.search(r"\b(?:listed|offered|asking|priced?|now)\b[^.!?]{0,30}\$\s*\d", lower) and not re.search(r"\b(?:rent|upgrades?|closing costs?|income)\b", lower):
                roles.add("price")
            if re.search(r"\b\d[\d,.]*[\s-]*(?:sq\.?\s*ft\.?|sqft|square[\s-]*(?:foot|feet)|acres?)\b", lower):
                roles.add("size")
            if re.search(r"\b(?:single.story|two.story|townhome|townhouse|condo|cabin|duplex|detached)\b", lower):
                roles.add("property_type")
            if re.search(r"\b(?:requires? repair|needs? repair|no permits|unpermitted|cash only|rehab financing)\b", lower):
                roles.add("condition_caution")
            terms = {item["term"] for item in features if item["sentence_index"] == index}
            if re.search(r"\b(?:pool|spa|fireplace|garage|solar|views?|adu|patio|deck)\b", lower):
                # Additional source words improve salience, but never create
                # unsupported canonical amenity labels in the returned metadata.
                terms |= set(re.findall(r"\b(?:fireplace|solar|adu)\b", lower))
            roles |= {"feature:" + term for term in terms}
            weights = {"beds": 4.5, "baths": 4.5, "price": 3.0, "location": 2.0,
                       "size": 1.5, "property_type": 1.0, "condition_caution": 4.0}
            base = sum(weights.get(role, 0) for role in roles)
            base += sum(sorted((feature_weights.get("feature:" + term, 1.5) for term in terms), reverse=True)[:2])
            if rooms["beds"] and rooms["baths"]:
                base += 3
            if re.search(r"\b(?:additional|secondary|guest|upstairs|rear unit)\b", lower) and not (rooms["beds"] and rooms["baths"]):
                base -= 1
            for key, alias in (("beds", "bedrooms"), ("baths", "bathrooms")):
                expected = self._number(entities.get(alias))
                if expected is not None and expected in rooms[key]:
                    base += .7
            boilerplate = bool(re.search(r"\b(?:call today|schedule|do not disturb|buyer to verify|go direct|ask agent|subject to lender|https?:)\b", lower))
            if boilerplate:
                base -= 5
            word_count = len(sentence.split())
            if word_count < 6 and not rooms["beds"] and not rooms["baths"]:
                base -= 2
            base += .5 / (index + 1)
            base -= max(0, word_count - 65) * .045
            candidates.append({"index": index, "text": sentence, "roles": sorted(roles),
                               "score": base, "word_count": word_count,
                               "boilerplate": boilerplate,
                               "tokens": set(re.findall(r"\w+", lower))})
        available_roles = sorted({role for row in candidates for role in row["roles"]})
        selected, covered = [], set()
        while candidates and len(selected) < num_sentences:
            ranking = []
            for candidate in candidates:
                roles = set(candidate["roles"])
                new = roles - covered
                gain = sum({"beds": 4.5, "baths": 4.5, "price": 3, "location": 2,
                            "size": 1.5, "property_type": 1, "condition_caution": 4}.get(role, 0) for role in new)
                already_features = sum(role.startswith("feature:") for role in covered)
                new_features = sorted((feature_weights.get(role, 1.5) for role in new if role.startswith("feature:")), reverse=True)
                gain += sum(new_features[:max(0, 2 - already_features)])
                similarities = [len(candidate["tokens"] & other["tokens"]) / max(1, len(candidate["tokens"] | other["tokens"])) for other in selected]
                redundancy = max(similarities, default=0)
                if redundancy > .8:
                    continue
                objective = gain + .35 * candidate["score"] - 2 * redundancy
                # Keep a soft word budget; never truncate or rewrite a sentence.
                total_words = sum(row["word_count"] for row in selected) + candidate["word_count"]
                objective -= max(0, total_words - 105) * .04
                ranking.append((objective, gain, candidate))
            if not ranking:
                break
            objective, gain, winner = max(ranking, key=lambda row: (row[0], -row[2]["index"]))
            if selected and winner["boilerplate"]:
                break
            if len(selected) >= 2 and (gain < 1 or objective <= 0):
                break
            selected.append(winner)
            covered |= set(winner["roles"])
            candidates.remove(winner)
        selected.sort(key=lambda row: row["index"])
        return {"summary": " ".join(row["text"] for row in selected),
                "selected_sentences": [{key: value for key, value in row.items() if key not in {"tokens", "boilerplate"}} for row in selected],
                "covered_roles": sorted(covered), "available_roles": available_roles,
                "source_sentence_count": len(sentences)}

    def extractive_summary(self, remarks: Any, entities: Mapping | None = None,
                           num_sentences: int = 2) -> str:
        return self.extractive_details(remarks, entities, num_sentences)["summary"]

    @classmethod
    def _first(cls, sources, keys):
        for source in sources:
            for key in keys:
                value = source.get(key)
                if cls.text(value):
                    return value
        return None

    @staticmethod
    def _number(value, *, allow_zero=False):
        try:
            result = float(str(value).replace(",", "").replace("$", ""))
            if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
                return None
            return int(result) if result.is_integer() else result
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _noun_phrase(term):
        if term in {"single-story", "two-story", "open-concept", "open floor plan"}:
            return "an open floor plan" if "open" in term else "a " + term + " layout"
        if term in {"updated", "recently updated", "renovated", "remodeled", "move-in ready"}:
            return term + " condition"
        if term == "ev charging":
            return "EV charging"
        if term.endswith("s") or any(word in term for word in ("flooring", "lighting", "light", "conditioning", "parking", "storage", "landscaping")):
            return term
        return ("an " if term[:1] in "aeiou" else "a ") + term

    def _template_listing(self, listing_record: Mapping[str, Any]) -> dict[str, Any]:
        """Legacy field template for comparison only; never the main summary."""
        if not isinstance(listing_record, Mapping):
            raise TypeError("listing_record must be a mapping")
        attributes = listing_record.get("listing_attributes") or {}
        sources = [attributes, listing_record]
        facts = {
            "beds": self._number(self._first(sources, ("listed_bedrooms", "beds", "L_Keyword2")), allow_zero=True),
            "baths": self._number(self._first(sources, ("listed_bathrooms", "baths", "LM_Dec_3")), allow_zero=True),
            "price": self._number(self._first(sources, ("listed_price", "price", "L_SystemPrice"))),
            "location": self.text(self._first(sources, ("city", "L_City"))) or None,
        }
        remarks = self.text(self._first(sources, ("remarks", "L_Remarks")))
        features = self.feature_evidence(remarks)[:2]
        first = "This home"
        if facts["location"]:
            first += " in " + facts["location"]
        rooms = []
        if facts["beds"] is not None:
            rooms.append(f"{facts['beds']:g} bedrooms")
        if facts["baths"] is not None:
            rooms.append(f"{facts['baths']:g} bathrooms")
        if rooms:
            first += " has " + " and ".join(rooms)
        if facts["price"] is not None:
            price = f"${facts['price']:,.0f}" if float(facts['price']).is_integer() else f"${facts['price']:,.2f}"
            first += (" and is listed at " if rooms else " is listed at ") + price
        else:
            first += ("; no listed price is available" if rooms else " has no listed price in the available record")
        first += "."
        second = "Highlights include " + " and ".join(self._noun_phrase(x["term"]) for x in features) + "." if features else "No supported feature highlights were identified in the available remarks."
        warnings = []
        # Flag explicit early room-count discrepancies for verification; do not
        # replace MLS totals using ambiguous secondary-bedroom mentions.
        for key, label in (("beds", r"bedrooms?|beds?"), ("baths", r"bathrooms?|baths?")):
            match = re.search(r"(?<![\w/])(\d+(?:\.\d+)?)(?:\s+(\d+)\s*/\s*(\d+))?\s*-?\s*(?:" + label + r")\b", remarks[:700], re.I)
            if match:
                mentioned = float(match.group(1))
                if match.group(2) and float(match.group(3)):
                    mentioned += float(match.group(2)) / float(match.group(3))
                if facts[key] is not None and mentioned != facts[key]:
                    warnings.append(f"{key}: listing field {facts[key]} differs from early remarks mention {mentioned:g}; verify total versus partial counts")
        summary = first + " " + second
        if warnings:
            summary += " Verify room counts: listing fields and remarks contain different counts."
        missing = [key for key, value in facts.items() if value is None]
        missing += [f"feature_{i + 1}" for i in range(len(features), 2)]
        return {"listing_id": self.text(self._first(sources, ("listing_id", "L_ListingID"))),
                "summary": summary, "method": "structured_plus_extractive",
                "extractive_summary": "",
                "facts": facts, "top_features": [item["term"] for item in features],
                "feature_evidence": features, "missing_fields": missing,
                "warnings": warnings, "fact_source": "listing_fields"}

    def summarize_listing(self, listing_record: Mapping[str, Any]) -> dict[str, Any]:
        """Primary output is copied source sentences; MLS facts remain separate."""
        context = self._template_listing(listing_record)
        remarks = self.text(self._first([listing_record], ("remarks", "L_Remarks")))
        extractor = self.signals.entity_extractor
        entities = {"bedrooms": extractor.extract_bedrooms(remarks),
                    "bathrooms": extractor.extract_bathrooms(remarks),
                    "city": context["facts"]["location"]}
        details = self.extractive_details(remarks, entities, num_sentences=3)
        source_roles = set(details["covered_roles"])
        selected_text = details["summary"]
        selected_features = self.feature_evidence(selected_text)[:2]
        missing = [key for key in ("beds", "baths", "price", "location") if key not in source_roles]
        missing += [f"feature_{i+1}" for i in range(len(selected_features), 2)]
        return {**context, "summary": selected_text, "extractive_summary": selected_text,
                "template_summary": context["summary"], "method": "extractive",
                "selected_sentences": details["selected_sentences"],
                "remarks_entities": entities, "top_features": [item["term"] for item in selected_features],
                "feature_evidence": selected_features, "missing_fields": missing,
                "not_in_remarks": [key for key in ("beds", "baths", "price", "location") if key not in details["available_roles"]],
                "available_roles": details["available_roles"],
                "summary_source": "remarks_only", "fact_source": "listing_fields_separate_from_summary"}
