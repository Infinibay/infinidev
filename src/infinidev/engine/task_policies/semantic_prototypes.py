"""Versioned semantic prototypes kept separate from prompt policy contracts."""

from __future__ import annotations

from dataclasses import dataclass


PROTOTYPE_SET_VERSION = "task-policy-semantic-prototypes-v3"


@dataclass(frozen=True)
class SemanticPrototypeSet:
    """Positive and hard-negative examples for one contrastive policy."""

    positive: tuple[str, ...]
    negative: tuple[str, ...]


PROTOTYPES: dict[str, SemanticPrototypeSet] = {
    "review.read_only": SemanticPrototypeSet(
        positive=(
            "revisa el PR pero no cambies nada",
            "review this patch only",
            "inspect the existing code and report concrete defects without applying fixes",
            "assess this module for regressions and leave the implementation untouched",
            "audita el código y entrega hallazgos priorizados sin corregirlos",
            "inspecciona el módulo y escribe un informe; no edites la implementación",
            "inspect the source for flaws and give me a prioritized report; leave it untouched",
            "review the pull request for regressions and report evidence without pushing a patch",
            "audit the authorization diff and rank concrete risks; make no edits",
            "inspect the parser changes and return findings only",
            "look for compatibility defects in this branch but leave every file unchanged",
            "revisa el cambio de esquema y enumera riesgos sin implementar soluciones",
            "audita el worker y señala fallos reproducibles; no modifiques el repositorio",
            "inspecciona este diff y prioriza hallazgos comprobables, solo lectura",
            "revise o patch e relate regressões sem aplicar correções",
            "audite a fronteira de permissões e não altere os arquivos",
            "examine le correctif et classe les défauts sans toucher au code",
            "relis la branche et fournis uniquement un rapport de risques vérifiés",
            "prüfe den Patch auf belegbare Regressionen und ändere nichts",
            "esamina la modifica e riporta difetti concreti senza correggerli",
        ),
        negative=(
            "review and fix the failures",
            "implement the requested change and then review it",
            "investigate several external alternatives and cite sources",
            "corrige los defectos encontrados en el módulo",
            "review the failures and implement every necessary correction",
            "inspect the patch, then commit and push the fixes",
            "find the bug and repair the implementation",
            "add the requested endpoint and verify it end to end",
            "compare database engines using primary sources",
            "write a migration plan with rollback steps",
            "summarize what this module currently does",
            "revisa y corrige todos los fallos que encuentres",
            "inspecciona el PR y aplica un parche mínimo",
            "investiga alternativas externas y recomienda una con fuentes",
            "implementa la nueva capacidad y después revísala",
            "revise e corrija os defeitos encontrados",
            "examine puis répare les régressions de cette branche",
            "prüfe den Code und behebe anschließend die Fehler",
            "esamina e correggi i problemi del modulo",
            "measure the endpoint latency and optimize the bottleneck",
        ),
    ),
    "bugfix.root_cause": SemanticPrototypeSet(
        positive=(
            "corrige este error",
            "fix the crash at startup",
            "make the existing failing behavior work correctly again",
            "find the defect behind this regression and repair it",
            "el comportamiento existente da un resultado incorrecto; arréglalo",
            "encuentra la causa del fallo reproducible y haz que vuelva a funcionar",
            "the retry scheduler waits after its final allowed attempt; restore the limit",
            "a timeout returns stale data instead of the documented error; repair the contract",
            "el backoff ejecuta un intento de más y rompe el límite configurado",
            "la espera continúa cuando ya no quedan intentos; recupera el comportamiento correcto",
            "un temporizador pierde el último evento al vencer; restablece la garantía existente",
            "the decoder now rejects a payload it accepted last release; restore compatibility",
            "a cancelled job still writes its result; repair the existing cancellation guarantee",
            "pagination duplicates the boundary record; make the established cursor contract hold",
            "the parser crashes on an empty but valid document; restore the promised behavior",
            "la caché devuelve datos vencidos después de invalidar; corrige la regresión",
            "el cliente omite el último elemento de cada página; recupera el resultado esperado",
            "o worker confirma a mensagem antes de persistir; restaure a ordem garantida",
            "le parseur ignore désormais un champ valide; rétablis le comportement antérieur",
            "der Resolver liefert seit dem Update doppelte Einträge; stelle den Vertrag wieder her",
        ),
        negative=(
            "add a new capability that did not exist before",
            "restructure working code without changing its output",
            "inspect the defect but do not implement a correction",
            "añade soporte para un caso de uso nuevo",
            "the output is correct but too slow; profile and optimize it",
            "introduce a new retry strategy that has never been supported",
            "add a command that users have never had before",
            "reorganize the internals while preserving every output",
            "benchmark the healthy endpoint and reduce its p99 latency",
            "inspect the suspected defect but do not change code",
            "compare alternative libraries and recommend one with sources",
            "document the current error behavior without repairing it",
            "añade soporte para un formato que nunca fue aceptado",
            "simplifica el módulo sin cambiar resultados observables",
            "mide el throughput correcto y elimina el cuello de botella",
            "revisa el fallo y entrega hallazgos sin editar archivos",
            "adicione uma nova opção de configuração ao cliente",
            "réorganise le composant sans modifier son comportement",
            "füge eine neue Benutzerfunktion hinzu",
            "esamina il difetto senza applicare correzioni",
        ),
    ),
    "refactor.preserve_behavior": SemanticPrototypeSet(
        positive=(
            "refactoriza este módulo",
            "clean up without changing behavior",
            "make this code easier to understand while keeping its outputs identical",
            "improve the internal structure but preserve every external contract",
            "simplifica la estructura interna sin alterar lo que hace",
            "haz el módulo más mantenible conservando el comportamiento observable",
            "quiero que el módulo sea más fácil de mantener sin cambiar lo que hace",
            "split the large state machine into focused components with identical transitions",
            "remove duplication inside the serializer without changing bytes on the wire",
            "rename and reorganize private helpers while callers observe the same API",
            "extract the scheduling logic but preserve retries, timing, and errors",
            "simplifica el flujo interno manteniendo idénticas todas las salidas",
            "separa responsabilidades privadas sin modificar el contrato público",
            "elimina duplicación estructural conservando exactamente el comportamiento",
            "reorganize os módulos internos sem alterar nenhuma resposta observável",
            "extraia os helpers mantendo a mesma API e os mesmos efeitos",
            "sépare les responsabilités internes sans changer les résultats",
            "vereinfache die interne Struktur bei identischem Verhalten",
            "riorganizza gli helper privati senza cambiare l'output",
            "untangle the dependency graph while all existing tests and contracts remain stable",
        ),
        negative=(
            "the error says refactor required",
            "repair behavior that is currently incorrect",
            "add a new user-visible option",
            "explica el mensaje refactor required sin tocar archivos",
            "repair the incorrect result returned by the parser",
            "restore a retry contract that currently fails",
            "add support for a new authentication method",
            "measure the slow query and optimize its execution",
            "review the code and report findings without editing",
            "research alternative architectures and recommend one",
            "corrige la regresión que duplica registros",
            "añade una opción visible para los usuarios",
            "optimiza el endpoint a partir de una línea base",
            "revisa el PR sin modificar el código",
            "corrija o resultado incorreto do decoder",
            "ajoute une nouvelle commande au client",
            "mesure puis accélère le traitement sous charge",
            "behebe den Absturz beim Start",
            "aggiungi una capacità che oggi non esiste",
            "the log says refactor required; translate that sentence only",
        ),
    ),
    "feature.contract_first": SemanticPrototypeSet(
        positive=(
            "implementa soporte para",
            "add a new command",
            "give this component the ability to handle a new use case",
            "extend the program with a capability users do not have yet",
            "haz que el módulo pueda resolver este nuevo caso de uso",
            "añade una opción nueva disponible para los usuarios",
            "introduce jittered backoff when only fixed delays exist today",
            "permite configurar una estrategia de reintento nueva que hoy no existe",
            "add streaming export, which the service cannot perform today",
            "let callers authenticate with a new credential format",
            "create a command for restoring archived sessions",
            "support a new webhook event without changing existing events",
            "añade paginación inversa, un flujo todavía no disponible",
            "permite exportar el informe en un formato nuevo",
            "crea una opción para que usuarios pausen trabajos en curso",
            "adicione suporte a um novo tipo de mensagem",
            "permita que clientes configurem uma política ainda inexistente",
            "ajoute un mode hors ligne que le produit ne propose pas encore",
            "ermögliche einen neuen bisher nicht unterstützten Ablauf",
            "aggiungi un formato di output che gli utenti non hanno ancora",
        ),
        negative=(
            "repair an existing behavior that regressed",
            "restructure the code while preserving all behavior",
            "describe how to use the existing command in the README",
            "corrige el resultado incorrecto actual",
            "restore an existing retry limit that now waits one attempt too long",
            "repair timeout handling that returns stale data",
            "fix the crash in an already supported command",
            "restore pagination behavior that regressed last release",
            "restructure the implementation without any visible change",
            "profile the existing endpoint and reduce latency",
            "review this proposed capability without implementing it",
            "research whether the capability is worth building",
            "corrige el formato existente que ahora produce datos inválidos",
            "refactoriza el cliente conservando todas sus capacidades",
            "mide y acelera el flujo que ya funciona correctamente",
            "explica cómo se usa la función actual sin cambiarla",
            "corrija a regressão no recurso já suportado",
            "répare le comportement existant qui ne fonctionne plus",
            "vereinfache den Code ohne neue Funktion einzuführen",
            "correggi il crash della funzione esistente",
        ),
    ),
    "research.evidence_first": SemanticPrototypeSet(
        positive=(
            "investiga las alternativas",
            "research the root cause first",
            "compare the available approaches and support the recommendation with sources",
            "gather reliable evidence and deliver a report before choosing a direction",
            "compara las opciones disponibles y recomienda una citando evidencia",
            "reúne información fiable y entrega un informe con hechos e inferencias separados",
            "investigate competing queue designs and cite primary benchmarks before recommending",
            "determine why the ecosystem chose this protocol using authoritative sources",
            "compare maintained libraries, licensing, and compatibility before advising",
            "gather evidence about deployment approaches and identify unresolved questions",
            "investiga alternativas de almacenamiento y sustenta la recomendación con fuentes",
            "compara implementaciones existentes separando hechos de inferencias",
            "averigua la causa documentada del cambio y entrega un informe con evidencia",
            "pesquise opções de mensageria e recomende uma com fontes primárias",
            "compare bibliotecas mantidas e registre riscos ainda incertos",
            "étudie les architectures possibles et cite les documents officiels",
            "compare les solutions disponibles avant de conseiller une direction",
            "untersuche die Alternativen anhand primärer Quellen",
            "confronta gli approcci esistenti e motiva la raccomandazione con prove",
            "survey the standards and produce an evidence-backed recommendation without editing code",
        ),
        negative=(
            "inspect this patch for concrete code defects",
            "implement the chosen approach now",
            "fix the reproducible failure in the current code",
            "revisa este archivo y lista defectos sin buscar fuentes externas",
            "implement the already selected approach now",
            "repair the failing parser and add a regression test",
            "refactor the module while preserving behavior",
            "review this diff for concrete defects only",
            "benchmark the endpoint and optimize the measured bottleneck",
            "write documentation for the existing command",
            "implementa la alternativa que ya elegimos",
            "corrige el fallo reproducible del worker",
            "revisa el parche sin consultar fuentes externas",
            "optimiza la latencia del servicio bajo carga",
            "implemente agora a solução já definida",
            "corrige le bug actuel et vérifie la régression",
            "prüfe den Patch auf konkrete Fehler ohne externe Recherche",
            "implementa subito l'approccio già deciso",
            "the issue says research required; explain that phrase only",
            "summarize the source file that is already open",
        ),
    ),
    "performance.measure_first": SemanticPrototypeSet(
        positive=(
            "optimiza esta función",
            "reduce request latency",
            "measure where this endpoint spends time and make it faster",
            "find the throughput bottleneck and improve the measured baseline",
            "mide dónde se consume el tiempo y acelera la operación",
            "encuentra el cuello de botella y mejora el rendimiento medido",
            "the result is correct but uses too much CPU; profile it and reduce the cost",
            "el resultado es correcto pero consume demasiada CPU; perfílalo y aceléralo",
            "benchmark cold startup and reduce the measured initialization time",
            "profile memory allocations in the working parser and lower the peak",
            "measure queue throughput under realistic load and remove the bottleneck",
            "reduce database query p95 after establishing a repeatable baseline",
            "mide el consumo de memoria del indexador y reduce el pico comprobado",
            "perfila el arranque correcto pero lento y mejora su tiempo medido",
            "aumenta el throughput del worker basándote en un benchmark reproducible",
            "meça as alocações do parser e reduza o custo observado",
            "otimize a latência p99 depois de medir a carga real",
            "mesure la consommation mémoire puis réduis le pic observé",
            "miss den Durchsatz unter Last und beseitige den Engpass",
            "misura il tempo di avvio e ottimizza il percorso critico",
        ),
        negative=(
            "clean up the code without changing behavior",
            "repair an incorrect result",
            "add a new feature without a performance goal",
            "document the current performance characteristics",
            "the retry loop waits after its last attempt and violates the configured limit",
            "a timeout returns an incorrect value instead of the documented error",
            "el backoff hace un intento extra y rompe el contrato existente",
            "la espera continúa después de cancelar y produce un resultado incorrecto",
            "un temporizador pierde eventos y rompe el resultado prometido",
            "fix the timeout that returns an incorrect status",
            "restore the retry limit that regressed",
            "add a new capability without a performance requirement",
            "restructure the module while preserving runtime characteristics",
            "review the benchmark patch without changing code",
            "research performance techniques and write a report only",
            "corrige el crash provocado por un deadline vencido",
            "añade un nuevo modo de exportación sin objetivo de velocidad",
            "refactoriza el flujo manteniendo el mismo rendimiento",
            "répare le résultat incorrect produit après un délai",
            "füge eine neue Funktion ohne Leistungsziel hinzu",
        ),
    ),
}


_AUGMENTATION_OBJECTS = ("AsterParser", "BirchQueue", "CinderClient", "DeltaWorker")
_POSITIVE_AUGMENTATION_TEMPLATES: dict[str, tuple[str, ...]] = {
    "bugfix.root_cause": (
        "{object} violates an established boundary on one valid input; restore the existing guarantee.",
        "A cancellation regression in {object} persists a result that must be discarded; repair it.",
        "The latest release made {object} return stale state for a supported request; recover the prior contract.",
        "{object} ejecuta una operación extra después del límite configurado; corrige la regresión.",
        "Una entrada válida provoca ahora un resultado incorrecto en {object}; restaura el comportamiento.",
        "{object} deixou de respeitar a ordem garantida; corrija a causa da regressão.",
        "{object} ne respecte plus un contrat existant; répare le défaut qui l'a rompu.",
    ),
    "feature.contract_first": (
        "Add a new user workflow to {object} that the current product cannot perform.",
        "Let callers use a credential format that {object} has never supported.",
        "Give {object} a new export mode while preserving every existing mode.",
        "Añade a {object} una capacidad solicitada que todavía no existe.",
        "Permite que {object} admita un flujo nuevo para sus usuarios.",
        "Adicione a {object} uma opção que hoje não está disponível.",
        "Ajoute à {object} un nouveau parcours sans modifier les capacités existantes.",
    ),
    "refactor.preserve_behavior": (
        "Split the coupled internals of {object} while every observable result remains identical.",
        "Extract focused helpers from {object} without changing its public contract.",
        "Reduce structural duplication in {object}; preserve outputs, errors, and side effects.",
        "Reorganiza las responsabilidades de {object} sin alterar ningún comportamiento visible.",
        "Simplifica el interior de {object} manteniendo idénticas API y salidas.",
        "Separe os módulos internos de {object} sem mudar o comportamento.",
        "Clarifie la structure de {object} sans aucun changement observable.",
    ),
    "research.evidence_first": (
        "Compare credible approaches for {object} using primary sources before recommending one.",
        "Gather authoritative evidence about {object} and distinguish facts from assumptions.",
        "Survey maintained options for {object}, including compatibility and licensing evidence.",
        "Investiga alternativas para {object} y sustenta la recomendación con fuentes fiables.",
        "Compara opciones de {object}, señalando explícitamente las incertidumbres.",
        "Pesquise soluções para {object} e recomende uma com evidências primárias.",
        "Étudie les choix possibles pour {object} à partir de documents officiels.",
    ),
    "review.read_only": (
        "Audit {object} for demonstrable defects and return findings without editing files.",
        "Review the {object} patch, rank concrete risks, and leave the branch untouched.",
        "Inspect {object} in read-only mode and report evidence rather than applying fixes.",
        "Revisa {object} y entrega solo hallazgos priorizados; no cambies el código.",
        "Audita el diff de {object} sin implementar ninguna corrección.",
        "Revise {object} sem alterar arquivos e relate problemas comprovados.",
        "Examine {object} sans modification et classe les défauts vérifiables.",
    ),
    "performance.measure_first": (
        "Benchmark {object} under representative load and reduce the measured bottleneck.",
        "Profile the correct but slow path in {object}, then improve its observed p95.",
        "Measure allocations in {object} and lower the demonstrated memory peak.",
        "Mide la latencia real de {object} y optimiza el camino crítico comprobado.",
        "Perfila {object}, que funciona bien pero consume demasiada CPU, y reduce el coste.",
        "Meça o throughput de {object} e elimine o gargalo observado.",
        "Mesure les performances de {object} puis optimise le coût démontré.",
    ),
}


def _expanded_prototypes() -> dict[str, SemanticPrototypeSet]:
    base = PROTOTYPES
    expanded: dict[str, SemanticPrototypeSet] = {}
    for policy_id, prototypes in base.items():
        positive = list(prototypes.positive)
        positive.extend(
            template.format(object=object_name)
            for template in _POSITIVE_AUGMENTATION_TEMPLATES[policy_id]
            for object_name in _AUGMENTATION_OBJECTS
        )
        negative = list(dict.fromkeys(prototypes.negative))
        for other_policy, other_prototypes in base.items():
            if other_policy == policy_id:
                continue
            for example in other_prototypes.positive:
                if example not in negative and example not in positive:
                    negative.append(example)
                if len(negative) >= 64:
                    break
            if len(negative) >= 64:
                break
        expanded[policy_id] = SemanticPrototypeSet(
            positive=tuple(dict.fromkeys(positive)),
            negative=tuple(negative[:64]),
        )
    return expanded


PROTOTYPES = _expanded_prototypes()


__all__ = ["PROTOTYPES", "PROTOTYPE_SET_VERSION", "SemanticPrototypeSet"]
