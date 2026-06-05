# AI Agent per copywriting di un blog finanziario

Questo pacchetto Python utilizza LangGraph per gestire un agente nella stesura di articoli in ambito finanziario. Il pacchetto sfrutta il modello gratuito llama-3.3-70b-versatile tramite LangChain_Groq per generare vari elementi dell'output. 

## Struttura

Il task dell'agente è stato fattorizzato in più nodi in modo da diminuire le probabilità di allucinazione dell'agente punto per punto. In particolare, il grafo è stato diviso nel seguente modo:

1. keyword_fetcher_node: questo nodo è preposto alla generazione delle parole chiave. Le prime 3 keyword vengono ricercate tramite SerpAPI. Le ultime due vengono invece generate invocando llama. Questa scelta è stata fatta per evitare che le keyword fossero tutte troppo meccaniche, ma comunque correlate all'argomento. Inoltre, in questa fase, l'LLM che viene invocata ha anche il compito di assicurarsi della pertinenza del topic con il mondo della finanza. Se il topic viene considerato non valido il codice si arresta generando un avviso;

2. outliner_node: una volta che le keyword sono state generate, viene richiamato un outliner che si occupa di costruire lo scheletro dell'articolo. Questo nodo utilizza llama per generare da 3 a 5 titoli per delle sezioni H2, basandosi sul topic e le keyword. Genera inoltre delle direttive per ogni sezione dove istruisce il copywriter in maniera sintetica sul contenuto della sezione. Questo è stato utilizzato per lo più allo scopo di evitare che il modello ricopiasse esattamente le keyword all'interno del testo. In fase di test si era provato a fare prompt engineering facendo esempi e dando divieti espliciti, ad esempio "è VIETATO utilizzare le keyword copiandole e incollandole nel testo e forzarle all'interno del testo. Aggiungi o rimuovi punteggiatura, altre parole, o inverti l'ordine delle parole all'interno di una keyword cercando di far mantenere un senso alla frase e rispettando la grammatica italiana (es: Cessione del quinto dello stipendio: simulazione -> Per simulare la cessione del quinto dello stipendio...)". Malgrado il prompt sia stato modificato più volte, l'LLM continuava ad inserire le keyword a forza all'interno del testo. Per questo motivo ho deciso di eliminarle dal prompt del copywriter ed ho affidato all'outliner il compito di tradurle in direttive naturali.

3. copywriter_node: questo è il nodo che si occupa della stesura vera e propria dell'articolo, utilizzando il topic inserito dall'utente, gli header H2 e le direttive inserite dall'outliner. Qui è stata data come istruzione quella di generare circa 1000 parole, rispettando tutti i punti definiti dal task.

4. refiner_node: questo nodo si occupa di fare editing di quanto scritto dal copywriter, taglia le parti ridondanti e cerca di utilizzare un linguaggio più formale. Si è scelto di fare scrivere 1000 parole al copywriter in modo da massimizzare le probabilità che il refiner non tagliasse troppo, scendendo sotto la soglia delle 600 parole necessarie per la pubblicazione su installazione WordPress;

5. formatting_node: questo nodo si occupa dell'estrazione dei metadati per l'installazione WordPress. Genera una meta description di 155 caratteri ed estrae il titolo dal testo. In questo nodo, senza utilizzo di agenti, si controlla tramite regex che tutto abbia la formattazione richiesta per la pubblicazione e che il testo sia della lunghezza corretta. Inoltre si utilizza il pacchetto markdown per convertire il testo markdown generato dall'agente in testo HTML

6. publishing_node: quest'ultimo nodo si occupa della pubblicazione su WordPress locale. In particolare, viene utilizzato un nodo di routing per controllare che il formatting_node abbia dato il via libera alla pubblicazione: se questo non è stato dato l'agente si ferma e l'output viene semplicemente salvato in JSON in una cartella articoli; viceversa, se il nodo ha dato il via libera alla pubblicazione, quest'ultima viene effettuata tramite REST API.

7. Alla fine il risultato in JSON comunque viene salvato all'interno di una cartella articoli dentro la cartella di lavoro.

## API Keys e setup

Tutte le API key utilizzate sono state salvate nel file .env
Per utilizzare il pacchetto è sufficiente creare un environment python

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python agent.py
```
Per testare la pubblicazione in formato bozza su installazione WordPress locale, installare LocalWP, creare un nuovo sito locare premendo il tasto in basso a sinistra, ed inserire i dati richiesti (URL e Nome utente (NON EMAIL)) all'interno del file .env. Creare una application password andando alla bacheca admin del sito locale (cliccando su wp-admin). Cliccare su Utenti nella barra laterale e poi sul proprio nome utente. In basso si troverà la sezione Scegli password per l'applicazione, crearne una nuova e inserirla nel .env nel campo WP_APP_PASS.

## Limitazioni

Utilizzando Groq con API gratuita e SerpAPI sempre con tier gratuito l'agente ha un massimo di 250 richieste al giorno a SerpAPI e 100k token giornalieri con llama. Limiti non troppo stringenti in fase di test, ma che sicuramente non vanno bene in fase di produzione.
Inoltre si è deciso di limitare lo spazio degli argomenti a quelli economico-finanziari per una questione di allucinazione del modello e per poter dare un ruolo definito all'agente. Questo introduce un'altra limitazione che consiste nell'assenza di un sistema di RAG all'interno del modello. Questa mancanza fa si che spesso l'agente produca alcuni risultati non accurati, che si è sempre cercato di evitare tramite prompt engineering. Inserendo un database da cui l'agente può andare a leggere facendo RAG questo effetto dovrebbe sparire.
