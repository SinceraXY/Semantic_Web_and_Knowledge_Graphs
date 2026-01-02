package org.example;

import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.apache.jena.ontology.OntClass;
import org.apache.jena.ontology.OntModel;
import org.apache.jena.ontology.OntModelSpec;
import org.apache.jena.ontology.OntProperty;
import org.apache.jena.ontology.OntResource;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.ResultSet;
import org.apache.jena.query.ResultSetFormatter;
import org.apache.jena.rdf.model.InfModel;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.riot.RDFDataMgr;
import org.apache.jena.reasoner.rulesys.GenericRuleReasoner;
import org.apache.jena.reasoner.rulesys.Rule;

public class App {
    private static final String NS = "http://example.org/cs#";

    public static void main(String[] args) {
        String ontologyFile = "campus_ontology.ttl";
        String instancesFile = "campus_instances.ttl";
        String rulesFile = "campus.rules";

        Model base = ModelFactory.createDefaultModel();
        RDFDataMgr.read(base, ontologyFile);
        RDFDataMgr.read(base, instancesFile);

        System.out.println("=== Base Model loaded ===");
        System.out.println("Triples: " + base.size());

        OntModel ontModel = ModelFactory.createOntologyModel(OntModelSpec.OWL_MEM_RDFS_INF, base);

        System.out.println();
        System.out.println("=== Ontology: Classes (in namespace) ===");
        List<OntClass> classes = listNamedClassesInNamespace(ontModel, NS);
        for (OntClass c : classes) {
            System.out.println("- " + c.getLocalName());
        }

        System.out.println();
        System.out.println("=== Ontology: Properties (in namespace) ===");
        List<OntProperty> props = listNamedPropertiesInNamespace(ontModel, NS);
        for (OntProperty p : props) {
            System.out.println("- " + p.getLocalName());
        }

        System.out.println();
        System.out.println("=== Instances by Class (explicit) ===");
        for (OntClass c : classes) {
            List<OntResource> individuals = listIndividuals(ontModel, c);
            if (individuals.isEmpty()) {
                continue;
            }
            System.out.println("[" + c.getLocalName() + "]");
            for (OntResource ind : individuals) {
                System.out.println("  - " + ind.getLocalName());
            }
        }

        System.out.println();
        System.out.println("=== SPARQL Query (explicit): students and enrolled courses ===");
        runSelectQuery(base,
                "PREFIX cs: <" + NS + ">\n" +
                        "SELECT ?student ?course\n" +
                        "WHERE {\n" +
                        "  ?student a cs:Student .\n" +
                        "  ?student cs:enrolledIn ?course .\n" +
                        "}\n" +
                        "ORDER BY ?student ?course\n");

        System.out.println();
        System.out.println("=== SPARQL Query (ontology): class hierarchy (owl:Class + rdfs:subClassOf) ===");
        runSelectQuery(base,
                "PREFIX cs: <" + NS + ">\n" +
                        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n" +
                        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" +
                        "SELECT ?class ?super\n" +
                        "WHERE {\n" +
                        "  ?class a owl:Class .\n" +
                        "  FILTER(STRSTARTS(STR(?class), STR(cs:)))\n" +
                        "  OPTIONAL {\n" +
                        "    ?class rdfs:subClassOf ?super .\n" +
                        "    FILTER(STRSTARTS(STR(?super), STR(cs:)))\n" +
                        "  }\n" +
                        "}\n" +
                        "ORDER BY ?class ?super\n");

        System.out.println();
        System.out.println("=== SPARQL Query (ontology): properties with domain/range ===");
        runSelectQuery(base,
                "PREFIX cs: <" + NS + ">\n" +
                        "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n" +
                        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" +
                        "SELECT ?prop ?type ?domain ?range\n" +
                        "WHERE {\n" +
                        "  ?prop a ?type .\n" +
                        "  FILTER(?type IN (owl:ObjectProperty, owl:DatatypeProperty))\n" +
                        "  FILTER(STRSTARTS(STR(?prop), STR(cs:)))\n" +
                        "  OPTIONAL { ?prop rdfs:domain ?domain . }\n" +
                        "  OPTIONAL { ?prop rdfs:range ?range . }\n" +
                        "}\n" +
                        "ORDER BY ?prop\n");

        System.out.println();
        System.out.println("=== Reasoning with custom rules ===");
        InfModel inf = createInfModelWithRules(base, rulesFile);
        System.out.println("Triples after reasoning (including inferred): " + inf.size());

        System.out.println();
        System.out.println("=== SPARQL Query (inferred): who is taught by which teacher ===");
        runSelectQuery(inf,
                "PREFIX cs: <" + NS + ">\n" +
                        "SELECT ?student ?teacher\n" +
                        "WHERE {\n" +
                        "  ?student cs:isTaughtBy ?teacher .\n" +
                        "}\n" +
                        "ORDER BY ?student ?teacher\n");

        System.out.println();
        System.out.println("=== SPARQL Query (inferred): excellent students and scholarship ===");
        runSelectQuery(inf,
                "PREFIX cs: <" + NS + ">\n" +
                        "SELECT ?student ?flag\n" +
                        "WHERE {\n" +
                        "  ?student a cs:ExcellentStudent .\n" +
                        "  OPTIONAL { ?student cs:eligibleForScholarship ?flag . }\n" +
                        "}\n" +
                        "ORDER BY ?student\n");
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

    private static void runSelectQuery(Model model, String sparql) {
        Query q = QueryFactory.create(sparql);
        try (QueryExecution qe = QueryExecutionFactory.create(q, model)) {
            ResultSet rs = qe.execSelect();
            ResultSetFormatter.out(System.out, rs, q);
        }
    }

    private static InfModel createInfModelWithRules(Model base, String rulesFile) {
        File f = new File(rulesFile);
        String ruleUrl = f.getAbsoluteFile().toURI().toString();
        List<Rule> rules = Rule.rulesFromURL(ruleUrl);
        GenericRuleReasoner reasoner = new GenericRuleReasoner(rules);
        reasoner.setMode(GenericRuleReasoner.HYBRID);
        return ModelFactory.createInfModel(reasoner, base);
    }
}
