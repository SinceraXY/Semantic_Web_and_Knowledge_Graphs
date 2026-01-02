package org.example;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import org.apache.jena.ontology.OntClass;
import org.apache.jena.ontology.OntModel;
import org.apache.jena.ontology.OntModelSpec;
import org.apache.jena.ontology.OntProperty;
import org.apache.jena.ontology.OntResource;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;
import org.apache.jena.rdf.model.InfModel;
import org.apache.jena.rdf.model.Literal;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.reasoner.rulesys.GenericRuleReasoner;
import org.apache.jena.reasoner.rulesys.Rule;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

public class WebApp {
    private static final String NS = "http://example.org/cs#";
    private static final Gson GSON = new Gson();

    private static class QueryRequest {
        String sparql;
        String model;
    }

    private static class QueryResponse {
        List<String> vars;
        List<Map<String, String>> rows;
        Boolean booleanResult;
        String text;
        String error;
        long elapsedMs;
    }

    private static class SummaryResponse {
        long baseTriples;
        long inferredTriples;
        String ns;
        String ontology;
        String instances;
        String rules;
    }

    public static void main(String[] args) {
        int p = 4567;
        String portProp = System.getProperty("port");
        String envPort = System.getenv("PORT");
        String chosen = portProp != null && !portProp.isBlank() ? portProp : envPort;
        if (chosen != null && !chosen.isBlank()) {
            p = Integer.parseInt(chosen.trim());
        }

        String ontologyFile = System.getProperty("ontology", "campus_ontology.ttl");
        String instancesFile = System.getProperty("instances", "campus_instances.ttl");
        String rulesFile = System.getProperty("rules", "campus.rules");

        String ontologyAbs = new File(ontologyFile).getAbsolutePath();
        String instancesAbs = new File(instancesFile).getAbsolutePath();
        String rulesAbs = new File(rulesFile).getAbsolutePath();

        Model base = loadBaseModel(ontologyFile, instancesFile);
        InfModel inf = createInfModelWithRules(base, rulesFile);

        try {
            HttpServer server = HttpServer.create(new InetSocketAddress(p), 0);
            server.setExecutor(Executors.newFixedThreadPool(8));

            server.createContext("/", ex -> {
                if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                    writeCors(ex);
                    ex.sendResponseHeaders(204, -1);
                    ex.close();
                    return;
                }

                String path = ex.getRequestURI().getPath();
                if ("/".equals(path)) {
                    redirect(ex, "/index.html");
                    return;
                }
                if (path.startsWith("/api/")) {
                    writeJson(ex, 404, error("Not Found"));
                    return;
                }

                if ("/index.html".equals(path) || "/app.js".equals(path) || "/styles.css".equals(path)) {
                    serveStatic(ex, path);
                    return;
                }

                writeJson(ex, 404, error("Not Found"));
            });

            server.createContext("/api/health", ex -> {
                if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                    writeCors(ex);
                    ex.sendResponseHeaders(204, -1);
                    ex.close();
                    return;
                }
                writeJson(ex, 200, Map.of("ok", true));
            });

            server.createContext("/api/summary", ex -> {
                if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                    writeCors(ex);
                    ex.sendResponseHeaders(204, -1);
                    ex.close();
                    return;
                }
                SummaryResponse out = new SummaryResponse();
                out.baseTriples = base.size();
                out.inferredTriples = inf.size();
                out.ns = NS;
                out.ontology = ontologyAbs;
                out.instances = instancesAbs;
                out.rules = rulesAbs;
                writeJson(ex, 200, out);
            });

            server.createContext("/api/ontology/classes", ex -> {
                if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                    writeCors(ex);
                    ex.sendResponseHeaders(204, -1);
                    ex.close();
                    return;
                }
                Map<String, String> qs = parseQuery(ex.getRequestURI().getRawQuery());
                Model m = chooseModel(qs.get("model"), base, inf);
                OntModel ont = ModelFactory.createOntologyModel(OntModelSpec.OWL_MEM_RDFS_INF, m);
                List<String> out = listNamedClassesInNamespace(ont, NS).stream().map(OntClass::getLocalName)
                        .collect(Collectors.toList());
                writeJson(ex, 200, out);
            });

            server.createContext("/api/ontology/properties", ex -> {
                if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                    writeCors(ex);
                    ex.sendResponseHeaders(204, -1);
                    ex.close();
                    return;
                }
                Map<String, String> qs = parseQuery(ex.getRequestURI().getRawQuery());
                Model m = chooseModel(qs.get("model"), base, inf);
                OntModel ont = ModelFactory.createOntologyModel(OntModelSpec.OWL_MEM_RDFS_INF, m);
                List<String> out = listNamedPropertiesInNamespace(ont, NS).stream().map(OntProperty::getLocalName)
                        .collect(Collectors.toList());
                writeJson(ex, 200, out);
            });

            server.createContext("/api/instances", ex -> {
                if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                    writeCors(ex);
                    ex.sendResponseHeaders(204, -1);
                    ex.close();
                    return;
                }
                Map<String, String> qs = parseQuery(ex.getRequestURI().getRawQuery());
                String classLocalName = qs.get("class");
                if (classLocalName == null || classLocalName.isBlank()) {
                    writeJson(ex, 400, error("Missing query param: class"));
                    return;
                }

                Model m = chooseModel(qs.get("model"), base, inf);
                OntModel ont = ModelFactory.createOntologyModel(OntModelSpec.OWL_MEM_RDFS_INF, m);
                OntClass c = ont.getOntClass(NS + classLocalName.trim());
                if (c == null) {
                    writeJson(ex, 404, error("Class not found: " + classLocalName));
                    return;
                }
                List<String> inds = listIndividuals(ont, c).stream().map(OntResource::getLocalName)
                        .collect(Collectors.toList());
                writeJson(ex, 200, inds);
            });

            server.createContext("/api/query", ex -> {
                if ("OPTIONS".equalsIgnoreCase(ex.getRequestMethod())) {
                    writeCors(ex);
                    ex.sendResponseHeaders(204, -1);
                    ex.close();
                    return;
                }
                if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                    writeJson(ex, 405, error("Method Not Allowed"));
                    return;
                }

                QueryRequest in;
                try {
                    String body = readBody(ex);
                    in = GSON.fromJson(body, QueryRequest.class);
                } catch (JsonSyntaxException e) {
                    writeJson(ex, 400, error("Invalid JSON body"));
                    return;
                }

                if (in == null || in.sparql == null || in.sparql.isBlank()) {
                    writeJson(ex, 400, error("sparql is required"));
                    return;
                }

                Model chosenModel = chooseModel(in.model, base, inf);
                String sparql = withDefaultPrefixes(in.sparql);

                long t0 = System.nanoTime();
                QueryResponse out = new QueryResponse();

                try {
                    Query q = QueryFactory.create(sparql);
                    if (q.isSelectType()) {
                        out.vars = new ArrayList<>();
                        out.rows = new ArrayList<>();
                        try (QueryExecution qe = QueryExecutionFactory.create(q, chosenModel)) {
                            ResultSet rs = qe.execSelect();
                            List<String> vars = rs.getResultVars();
                            out.vars.addAll(vars);
                            while (rs.hasNext()) {
                                QuerySolution sol = rs.next();
                                Map<String, String> row = new LinkedHashMap<>();
                                for (String v : vars) {
                                    RDFNode n = sol.get(v);
                                    row.put(v, formatNode(n));
                                }
                                out.rows.add(row);
                            }
                        }
                    } else if (q.isAskType()) {
                        try (QueryExecution qe = QueryExecutionFactory.create(q, chosenModel)) {
                            out.booleanResult = qe.execAsk();
                        }
                    } else {
                        out.error = "Only SELECT and ASK are supported in this UI";
                        out.elapsedMs = (System.nanoTime() - t0) / 1_000_000;
                        writeJson(ex, 400, out);
                        return;
                    }
                } catch (Exception e) {
                    out.error = e.getClass().getSimpleName() + ": " + e.getMessage();
                    out.elapsedMs = (System.nanoTime() - t0) / 1_000_000;
                    writeJson(ex, 400, out);
                    return;
                }

                out.elapsedMs = (System.nanoTime() - t0) / 1_000_000;
                writeJson(ex, 200, out);
            });

            server.start();
            System.out.println("Web UI started on http://localhost:" + p + "/ (model ns: " + NS + ")");
            System.out.println("Files: ontology=" + ontologyFile + ", instances=" + instancesFile + ", rules=" + rulesFile);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static Map<String, Object> error(String message) {
        return Map.of("error", message);
    }

    private static void redirect(HttpExchange ex, String location) throws IOException {
        Headers h = ex.getResponseHeaders();
        writeCors(ex);
        h.set("Location", location);
        ex.sendResponseHeaders(302, -1);
        ex.close();
    }

    private static void serveStatic(HttpExchange ex, String path) throws IOException {
        String resourcePath = "/web" + path;
        String contentType = guessContentType(path);
        byte[] data;
        try (InputStream is = WebApp.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                writeJson(ex, 404, error("Static resource not found"));
                return;
            }
            data = readAllBytes(is);
        }

        Headers h = ex.getResponseHeaders();
        writeCors(ex);
        h.set("Content-Type", contentType);
        ex.sendResponseHeaders(200, data.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(data);
        }
    }

    private static String guessContentType(String path) {
        if (path.endsWith(".html")) {
            return "text/html; charset=utf-8";
        }
        if (path.endsWith(".js")) {
            return "text/javascript; charset=utf-8";
        }
        if (path.endsWith(".css")) {
            return "text/css; charset=utf-8";
        }
        return "application/octet-stream";
    }

    private static void writeJson(HttpExchange ex, int status, Object payload) throws IOException {
        byte[] data = GSON.toJson(payload).getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        writeCors(ex);
        h.set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, data.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(data);
        }
    }

    private static void writeCors(HttpExchange ex) {
        Headers h = ex.getResponseHeaders();
        h.set("Access-Control-Allow-Origin", "*");
        h.set("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
        h.set("Access-Control-Allow-Headers", "Content-Type");
    }

    private static Map<String, String> parseQuery(String rawQuery) {
        Map<String, String> out = new LinkedHashMap<>();
        if (rawQuery == null || rawQuery.isBlank()) {
            return out;
        }
        String[] pairs = rawQuery.split("&");
        for (String pair : pairs) {
            if (pair.isEmpty()) {
                continue;
            }
            int idx = pair.indexOf('=');
            String k = idx >= 0 ? pair.substring(0, idx) : pair;
            String v = idx >= 0 ? pair.substring(idx + 1) : "";
            k = urlDecode(k);
            v = urlDecode(v);
            out.put(k, v);
        }
        return out;
    }

    private static String urlDecode(String s) {
        return URLDecoder.decode(s, StandardCharsets.UTF_8);
    }

    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream is = ex.getRequestBody()) {
            return new String(readAllBytes(is), StandardCharsets.UTF_8);
        }
    }

    private static byte[] readAllBytes(InputStream is) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = is.read(buf)) >= 0) {
            baos.write(buf, 0, n);
        }
        return baos.toByteArray();
    }

    private static Model loadBaseModel(String ontologyFile, String instancesFile) {
        Model base = ModelFactory.createDefaultModel();
        RDFDataMgr.read(base, ontologyFile);
        RDFDataMgr.read(base, instancesFile);
        return base;
    }

    private static InfModel createInfModelWithRules(Model base, String rulesFile) {
        File f = new File(rulesFile);
        String ruleUrl = f.getAbsoluteFile().toURI().toString();
        List<Rule> rules = Rule.rulesFromURL(ruleUrl);
        GenericRuleReasoner reasoner = new GenericRuleReasoner(rules);
        reasoner.setMode(GenericRuleReasoner.HYBRID);
        return ModelFactory.createInfModel(reasoner, base);
    }

    private static Model chooseModel(String model, Model base, InfModel inf) {
        if (model == null) {
            return base;
        }
        String m = model.trim().toLowerCase();
        if (Objects.equals(m, "inf") || Objects.equals(m, "inferred")) {
            return inf;
        }
        return base;
    }

    private static String withDefaultPrefixes(String sparql) {
        String s = sparql.trim();
        String lower = s.toLowerCase();

        StringBuilder prefixes = new StringBuilder();
        if (!lower.contains("prefix rdf:")) {
            prefixes.append("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n");
        }
        if (!lower.contains("prefix rdfs:")) {
            prefixes.append("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n");
        }
        if (!lower.contains("prefix owl:")) {
            prefixes.append("PREFIX owl: <http://www.w3.org/2002/07/owl#>\n");
        }
        if (!lower.contains("prefix xsd:")) {
            prefixes.append("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n");
        }
        if (!lower.contains("prefix cs:")) {
            prefixes.append("PREFIX cs: <" + NS + ">\n");
        }
        if (!lower.contains("prefix :")) {
            prefixes.append("PREFIX : <" + NS + ">\n");
        }

        if (prefixes.length() == 0) {
            return s;
        }
        return prefixes + s;
    }

    private static String formatNode(RDFNode n) {
        if (n == null) {
            return "";
        }
        if (n.isLiteral()) {
            Literal lit = n.asLiteral();
            return lit.getLexicalForm();
        }
        if (n.isResource()) {
            Resource r = n.asResource();
            if (r.getURI() == null) {
                return r.getId().getLabelString();
            }
            if (r.getURI().startsWith(NS)) {
                return "cs:" + r.getLocalName();
            }
            return r.getURI();
        }
        return n.toString();
    }

    private static List<OntClass> listNamedClassesInNamespace(OntModel m, String ns) {
        List<OntClass> out = new ArrayList<>();
        m.listClasses().forEachRemaining(c -> {
            if (c.getURI() == null) {
                return;
            }
            if (ns.equals(c.getNameSpace())) {
                out.add(c);
            }
        });
        out.sort(Comparator.comparing(OntClass::getLocalName));
        return out;
    }

    private static List<OntProperty> listNamedPropertiesInNamespace(OntModel m, String ns) {
        List<OntProperty> out = new ArrayList<>();
        m.listAllOntProperties().forEachRemaining(p -> {
            if (p.getURI() == null) {
                return;
            }
            if (ns.equals(p.getNameSpace())) {
                out.add(p);
            }
        });
        out.sort(Comparator.comparing(OntProperty::getLocalName));
        return out;
    }

    private static List<OntResource> listIndividuals(OntModel m, OntClass c) {
        List<OntResource> out = new ArrayList<>();
        m.listIndividuals(c).forEachRemaining(ind -> {
            if (ind.getURI() != null && NS.equals(ind.getNameSpace())) {
                out.add(ind);
            }
        });
        out.sort(Comparator.comparing(OntResource::getLocalName));
        return out;
    }

    private static String escapeJson(String s) {
        if (s == null) {
            return "";
        }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
