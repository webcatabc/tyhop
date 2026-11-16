package org.example.utils;

import java.text.Normalizer;
import java.util.Locale;
import java.util.Set;

/**
 * Canonicalizes entity names, relation names, and coarse entity-type buckets.
 * The type bucket deliberately groups ontology labels that describe the same
 * real-world kind (for example occupation/role/profession) so the same entity
 * cannot be split into several nodes only because the extractor chose a
 * different synonym.
 */
public final class EntityRelationNormalizer {
    private static final Set<String> OCCUPATION = Set.of(
            "profession", "occupation", "role", "position", "job");
    private static final Set<String> PERSON = Set.of(
            "person", "people", "individual", "human");
    private static final Set<String> LOCATION = Set.of(
            "location", "place", "city", "region", "country", "nation",
            "address", "area", "continent", "geography");
    private static final Set<String> ORGANIZATION = Set.of(
            "organization", "organisation", "company", "institution", "studio",
            "record_label", "label", "group", "team", "band", "music_band",
            "sports_team");
    private static final Set<String> DATE = Set.of(
            "date", "year", "time_period", "duration", "time", "period");
    private static final Set<String> QUANTITY = Set.of(
            "quantity", "rank", "integer", "number", "string", "count", "score");
    private static final Set<String> AWARD = Set.of("award", "honor", "prize");
    private static final Set<String> EVENT = Set.of(
            "event", "competition", "tournament", "sports_event");
    private static final Set<String> PUBLICATION = Set.of(
            "publication", "book", "novel", "magazine", "newspaper");

    private EntityRelationNormalizer() { }

    /** Stable identity form for an entity mention. */
    public static String normalizeName(String name) {
        if (name == null) return "";
        String value = Normalizer.normalize(name, Normalizer.Form.NFKC)
                .trim().toLowerCase(Locale.ROOT)
                .replaceAll("\\s+", " ");
        return value.replaceAll("^[\\p{Punct}\\s]+|[\\p{Punct}\\s]+$", "");
    }

    /**
     * Canonical relation form: lowercase, punctuation and whitespace become
     * single underscores. This turns "directed by" and "directed_by" into the
     * same relation key.
     */
    public static String normalizeRelation(String relation) {
        if (relation == null) return "";
        String value = normalizeName(relation).replaceAll("[^a-z0-9]+", "_");
        value = value.replaceAll("_+", "_").replaceAll("^_+|_+$", "");
        return value;
    }

    /** Coarse type bucket used as part of the entity identity. */
    public static String typeBucket(String type) {
        String value = normalizeRelation(type);
        if (value.isBlank()) return "unknown";
        if (OCCUPATION.contains(value)) return "occupation";
        if (PERSON.contains(value)) return "person";
        if (LOCATION.contains(value)) return "location";
        if (ORGANIZATION.contains(value)) return "organization";
        if (DATE.contains(value)) return "date";
        if (QUANTITY.contains(value)) return "quantity";
        if (AWARD.contains(value)) return "award";
        if (EVENT.contains(value)) return "event";
        if (PUBLICATION.contains(value)) return "publication";
        // Film/TV/music/book subkinds intentionally stay separate: a film,
        // book, and series can share a title without being the same entity.
        return value;
    }
}
