package org.example.entity;

import java.util.List;

/** Full type-mapping result for one constructed sample graph. */
public record TypeMappingResult(List<KnowledgeTriple> triples,
                                List<String> types) {
    public TypeMappingResult {
        triples = triples == null ? List.of() : List.copyOf(triples);
        types = types == null ? List.of() : List.copyOf(types);
    }
}
