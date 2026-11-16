package org.example.utils.json;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

/** Generic Gson conversion helpers for object/JSON mapping. */
public final class JsonCodec {
    private static final Gson GSON = new GsonBuilder().create();

    private JsonCodec() {
    }

    public static String toJson(Object value) {
        return GSON.toJson(value);
    }

    public static <T> T fromJson(String json, Class<T> type) {
        return GSON.fromJson(json, type);
    }

    public static <T> List<T> fromJsonList(String json, Class<T> elementType) {
        Type listType = TypeToken.getParameterized(List.class, elementType).getType();
        return GSON.fromJson(json, listType);
    }
}
