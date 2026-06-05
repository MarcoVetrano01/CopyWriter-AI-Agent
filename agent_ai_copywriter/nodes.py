import os
from typing import Any, Dict
import requests
from .agentstate import AgentState, Outline, ValidityCheckState, WordPressMetadata
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import base64
import markdown
import re

load_dotenv()  
modello = "llama-3.3-70b-versatile"
def keyword_fetcher_node(state: AgentState) -> AgentState:
    """
    Node to fetch keywords based on the topic provided by the user. This node will use the SerpAPI to get relevant keywords for the article.
    The fetched keywords will be added to the agent state and returned for further processing in the orchestration pipeline.
    """

    # Fetch keywords using SerpAPI
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError("SerpAPI key not found in environment variables. Set SERPAPI_KEY in your .env file.")
    
    topic = state.topic
    params = {
        "engine": "google",
        "q": topic,
        "api_key": api_key,
        "hl": "it",
        "gl": "it"
    }
    
    url = "https://serpapi.com/search"
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        keywords = []
        if 'related_searches' in data:
            keywords = [item['query'] for item in data['related_searches'][:3]]
        print(keywords)
        print("Invoking LLM to generate other keywords to be integrated in the article")

        SYSTEM_PROMPT = """Sei un Senior SEO Strategist specializzato nel Settore Bancario e Finanziario. Il tuo compito è analizzare un topic proposto e stabilire se ha senso logico e utilità reale scriverci un articolo per un blog finanziario.\n
                        CRITERI DI VALIDAZIONE:\n

                        Il topic deve riguardare un concetto finanziario, bancario, economico o fiscale reale.\n
                        Il topic non deve essere un'assurdità logica o una domanda impossibile.\n
                        Il topic deve avere un intento di ricerca chiaro e risolvibile (es. procedure, spiegazioni, confronti).\n

                        ESEMPI DI VALUTAZIONE:\n

                        Topic: 'Come insegno l'italiano al mio cane?' -> NON VALIDO (Motivo: Non ha senso logico ed è fuori dal dominio bancario).\n
                        Topic: 'Come si richiede un mutuo?' -> VALIDO (Motivo: Intento di ricerca chiaro e pertinente).\n
                        Topic: 'Come prelevare soldi da un bancomat spento' -> NON VALIDO (Motivo: Impossibile e incoraggia frodi).\n

                        REGOLE DI OUTPUT:\n

                        Se il topic è VALIDO: genera esattamente 5 parole chiave correlate ad alto volume di ricerca, e inseriscile nella lista delle keyword nel JSON. Non generare keyword che siano l'exact match del topic\n
                        Se il topic NON È VALIDO: lascia vuota la lista delle keyword e scrivi una breve motivazione (massimo 2 righe).\n

                        Rispondi ESCLUSIVAMENTE con l'output richiesto, senza alcun testo introduttivo o conclusivo."""

        llm_api_key = os.getenv("GROQ_API_KEY")
        if not llm_api_key:
            raise ValueError("Groq API key not found in environment variables. Set GROQ_API_KEY in your .env file.")
        llm = ChatGroq(model=modello, temperature=0.1, api_key = llm_api_key)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", f"Valuta il seguente topic: '{topic}'")
        ])
        structured_llm = llm.with_structured_output(ValidityCheckState)
        chain = prompt | structured_llm
        response = chain.invoke({"topic": topic})
        if not response.is_valid:
            print("Il topic è stato valutato come non valido. Motivo:", response.reason)
            state.keywords = []
            return state
        
        if not keywords:
            keywords = response.keywords
        else:
            keywords = keywords + response.keywords[:2]
            
        fetched = "generate"
        state.keywords = keywords
        print(f"Keywords {fetched}: {keywords}")
        return state
    except requests.exceptions.RequestException as e:
        print(f"Error fetching keywords: {e}")
        return state
    
def outliner_node(state: AgentState) -> list:
    """
    Node to generate an outline for the article based on the topic and keywords in the agent state. This node will use an LLM to create a structured outline that includes the main sections and subtopics to be covered in the article.
    The generated outline will be added to the agent state and returned for further processing in the orchestration pipeline.
    """

    SYSTEM_PROMPT = """Agisci come un Senior SEO Strategist specializzato nel settore Bancario e Finanziario.
                    Ricevi in input un Argomento e una lista di Keyword. Il tuo compito è creare la scaletta dell'articolo (Outline) e fornire istruzioni precise al copywriter su come gestire le keyword.
                    REGOLE PER I TITOLI (H2):
                    Crea da 3 a 5 sezioni logiche che coprano l'argomento (es. funzionamento, requisiti, simulazione, differenze INPS/NoiPA).
                    I titoli H2 devono essere naturali e discorsivi. È VIETATO usare le keyword crude come titoli.
                    REGOLE PER LE DIRETTIVE AL COPYWRITER:
                    Per ogni H2 che crei, devi scrivere una o più direttive (chiave copywriter_directive). In questa direttiva devi
                    dare delle istruzioni chiare e brevi su cosa scrivere nel paragrafo, riferendoti alle keyword assegnate a quella sezione, destrutturandone i concetti. TUTTE le keyword devono essere discusse nell'articolo in maniera naturale, 
                    quindi genera le sezioni e le direttive, basandoti su di esse (es. keyword: 'Cessione del quinto: Simulazione' ->direttiva: 'Spiega come funziona la simulazione della cessione del quinto, quali dati servono e quali risultati aspettarsi.').
                    FORMATO DI OUTPUT:
                    Rispondi ESCLUSIVAMENTE rispettando il formato richiesto dal JSON schema Outline, senza aggiungere testo introduttivo o conclusivo.
                    """
    
    llm_api_key = os.getenv("GROQ_API_KEY")
    if not llm_api_key:
        raise ValueError("Groq API key not found in environment variables. Set GROQ_API_KEY in your .env file.")
    llm = ChatGroq(model=modello, temperature=0.2, api_key = llm_api_key)
    structured_llm = llm.with_structured_output(Outline)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", f"Genera una lista di sezioni per l'articolo sul topic: '{state.topic}', considerando queste keyword: {', '.join(state.keywords)}.")
    ])
    chain = prompt | structured_llm
    response = chain.invoke({})
    
    return response.outline, response.copywriter_directives

def copywriter_node(state: AgentState) -> AgentState:
    """
    Node to generate the main content of the article based on the topic and keywords in the agent state. This node will use an LLM to create a 
    well-structured article that includes the specified keywords in a natural way.
    The generated content will be added to the agent state and returned for further processing in the orchestration pipeline.
    """

    SYSTEM_PROMPT = """
                    Agisci come un Senior SEO Copywriter e Tecnico Finanziario (settore YMYL).
                    Ricevi in input un Argomento (Topic), una Scaletta di sezioni (H2) e direttive su come sviluppare ogni sezione. Il tuo compito è scrivere il testo completo dell'articolo in formato Markdown.
                    REGOLE DI STRUTTURA E LUNGHEZZA:
                    Lunghezza: Il testo deve superare rigorosamente le 1000 parole. È VIETATO ripetere frasi o concetti per fare volume.
                    Formattazione: Inizia con un singolo titolo H1 (usando # Titolo). Usa ESATTAMENTE le sezioni fornite in input come titoli H2 (usando ## Titolo). Sotto ogni H2 scrivi il relativo paragrafo.
                    DENSITÀ TECNICA E QUALITÀ (YMYL):
                    Niente frasi introduttive scolastiche come "In questo articolo esploreremo...". Entra subito nel vivo del discorso fin dalla prima riga.
                    Evita di inventare informazioni o dettagli, numeri, leggi o statistiche. Se non sei sicuro di un dato, non inserirlo.
                    REGOLE SINTATTICHE E STILISTICHE (ANTI-ROBOT):
                    È SEVERAMENTE VIETATO ripetere lo stesso soggetto o la keyword principale in frasi consecutive. Usa i pronomi (es. "questa misura", "il finanziamento", "tale opzione") e varia la struttura dei periodi.
                    Non elencare concetti usando strutture meccaniche identiche (es. "Il requisito A è... Il requisito B è..."). Unisci i concetti in una prosa argomentata e coesa.
                    Titolo H1: Naturale, giornalistico e accattivante. Vietati i formati banali come "[Topic]: guida completa" o "[Topic]: cos'è e come funziona".
                    """
    if not state.keywords:
        print("Nessuna keyword disponibile per la generazione del contenuto. Il nodo copywriter non può procedere.")
        state.content = "Mi dispiace, ma non posso generare il contenuto dell'articolo perchè il topic è stato valutato come non pertinente."
        return state
    outline, copywriter_directives = outliner_node(state)
    llm_api_key = os.getenv("GROQ_API_KEY")
    if not llm_api_key:
        raise ValueError("Groq API key not found in environment variables. Set GROQ_API_KEY in your .env file.")
    llm = ChatGroq(model= modello, temperature=0.3, api_key = llm_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", f"Scrivi un articolo approfondito sul topic: '{state.topic}'. \nStruttura l'articolo utilizzando le seguenti sezioni: {', '.join(outline)}. Usa le direttive: {', '.join(copywriter_directives)}. ")
    ])
    chain = prompt | llm
    response = chain.invoke({})
    state.content = response.content    
    print("Contenuto generato dall'LLM.")
    return state

def refiner_node(state: AgentState) -> AgentState:
    """
    Node to refine the generated content of the article, checks the grammar and the overall quality of the text, ensuring that it is well-written and free of errors. 
    This node will use an LLM to analyze the content and make necessary improvements.
    """
    SYSTEM_PROMPT = """Agisci come un Editor SEO Spietato e Revisore Capo per il settore Bancario e Finanziario. Il tuo compito è prendere una bozza di articolo generata da un'IA di basso livello e trasformarla in un testo professionale, denso, umano e impeccabile.
                    REGOLE TASSATIVE DI REVISIONE:
                    Licenza di uccidere la fuffa: Elimina SPIETATAMENTE ogni frase ripetitiva, introduzioni banali ('in questo articolo esploreremo'), frasi fatte ('è importante notare che') e conclusioni che non aggiungono dati tecnici. Se un concetto è già stato espresso in un paragrafo precedente, CANCELLALO senza pietà. Il testo deve essere denso, non lungo.
                    Distruzione del Keyword Stuffing: Cerca nel testo qualsiasi keyword che suoni robotica, forzata o sgrammaticata. SMONTALA. Inserisci preposizioni, articoli, punteggiatura o inverti l'ordine delle parole per farla scorrere perfettamente in un italiano nativo e colloquiale.
                    Upgrade Lessicale: Sostituisci i termini base con il lessico tecnico bancario (es. quota cedibile, TAN, TAEG, delegazione di pagamento, trattenuta in busta paga).
                    Fluidità: Spezza i periodi troppo lunghi. Garantisci una transizione logica e fluida tra un H2 e l'altro.
                    Restituisci ESCLUSIVAMENTE l'articolo finale revisionato, formattato in Markdown. Non aggiungere alcun commento.
                    """  
    if state.keywords == []:
        return state     
    
    llm_api_key = os.getenv("GROQ_API_KEY")
    if not llm_api_key:
        raise ValueError("Groq API key not found in environment variables. Set GROQ_API_KEY in your .env file.")
    llm = ChatGroq(model=modello, temperature=0.1, api_key = llm_api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", f"Rivedi e migliora questo articolo, mantenendo intatto il significato originale:\n\n{state.content}")
    ])
    chain = prompt | llm
    response = chain.invoke({})
    state.content = response.content
    print("Contenuto raffinato dall'LLM.")                  

def formatting_node(state: AgentState) -> AgentState:
    """
    Node to format the generated content for WordPress. This node will ensure that the content is properly structured with HTML tags and ready to be published on a WordPress site.
    The formatted content will be added to the agent state and the wp_ready flag will be set to true if the content is ready for publication.
    """
    if state.keywords == []:
        return state

    SYSTEM_PROMPT = """Sei un esperto SEO. Il tuo unico compito è leggere l'articolo fornito dall'utente e generare:
                    1. Una meta description accattivante e informativa (massimo 155 caratteri).o lo schema JSON richiesto, senza aggiungere testo in
                    2. Estrapolare il titolo principale (se presente) o crearne uno ottimizzato SEO se assente.
                    RISPONDI SOLO RISPETTANDO LO SCHEMA JSON. NON INSERIRE L'ARTICOLO NELL'OUTPUT."""
    
    if state.keywords == []:
        return state
    llm_api_key = os.getenv("GROQ_API_KEY")
    if not llm_api_key:
        raise ValueError("Groq API key not found in environment variables. Set GROQ_API_KEY in your .env file.")
    llm = ChatGroq(model=modello, temperature=0.1, api_key = llm_api_key)

    match = re.search(r'^#\s+(.*)$', state.content, re.MULTILINE)
    extracted_title = match.group(1).strip() if match else state.topic
    clean_content = re.sub(r'^#\s+.*$\n*', '', state.content, count=1, flags=re.MULTILINE).strip()
    word_count = len(clean_content.split())
    h2_count = len(re.findall(r'^##\s+', clean_content, re.MULTILINE))

    is_ready = True
    if word_count < 600:
        print(f"Il contenuto ha solo {word_count} parole. Deve essere almeno 600.")
        is_ready = False
    if h2_count < 3:
        print(f"Il contenuto ha solo {h2_count} paragrafi con titoli H2. Deve essere almeno 3.")
        is_ready = False
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", f"Genera i metadati per questo articolo:\n\n{clean_content[:1500]}") 
    ])
    structured_llm = llm.with_structured_output(WordPressMetadata)
    chain = prompt | structured_llm
    response = chain.invoke({})
    state.title = response.title if response.title else extracted_title
    state.content = markdown.markdown(clean_content)
    state.meta_description = response.meta_description
    state.wp_ready = is_ready
    print("Contenuto formattato per WordPress.")
    return state

def publishing_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to publish the formatted content on WordPress. This node will use the WordPress REST API to create a new post with the 
    generated content.
    """

    if state.keywords == []:
        return {}
    wp_url = os.getenv("WP_URL") 
    wp_user = os.getenv("WP_USER")
    wp_app_password = os.getenv("WP_APP_PASS")

    endpoint = f"{wp_url}/wp-json/wp/v2/posts"
    credentials = f"{wp_user}:{wp_app_password}"
    token = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "title": state.title,
        "content": state.content,
        "status": "draft"
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        print("Articolo caricato su WP con successo!")
    except requests.exceptions.RequestException as e:
        print(f"Errore caricamento su WordPress: {e}")
        
    return {}

def routing_node(state: AgentState) -> str:
    """
    Node to route the agent workflow to the END or to the publishing node based on the wp_ready flag in the agent state.
    """

    if state.wp_ready:
        return "publish"
    else:
        return "end"