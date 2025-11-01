<h1>RAG-Powered Chatbot with Google Docs Integration</h1>

<h2>💡 Introduction</h2>

<p>
The <strong>RAG-Powered Chatbot with Google Docs Integration</strong> is an AI chatbot that combines the power of Retrieval-Augmented Generation (RAG) with Google Docs. It allows users to sign in with Google, select their documents, and ask questions in natural language. The chatbot retrieves relevant information from the selected documents and generates context-aware responses. If no information is found, it gracefully falls back to general knowledge responses.
</p>

<h2>🚀 Features</h2>

<ul>
  <li><strong>Google OAuth 2.0</strong> authentication for secure access</li>
  <li><strong>View & select Google Docs</strong> from your Drive</li>
  <li><strong>Document-aware Q&A:</strong> uses RAG pipeline to fetch context from selected docs</li>
  <li><strong>Fallback response:</strong> uses general LLM knowledge when document info is missing</li>
  <li><strong>Multilingual chat support:</strong> English, Hindi, French, German</li>
  <li><strong>PDF file upload and parsing</strong> support</li>
  <li><strong>Utility tools:</strong> Weather, Timezone, Currency, Unit converter, Wikipedia, Translator</li>
  <li><strong>YouTube transcript extraction</strong> support</li>
  <li><strong>Chat memory:</strong> thread-based chat history with session persistence</li>
  <li><strong>Export chat history</strong> as downloadable file</li>
</ul>

<h2>🖼️ Demo</h2>

<p>
<b>Live demo link:</b> <a href="#">(Coming Soon)</a><br/>
<b>Preview:</b><br/>
<img src="https://via.placeholder.com/800x400.png?text=Chatbot+Demo+Screenshot" alt="Demo Screenshot" />
</p>

<h2>⚙️ Installation & Setup</h2>

<pre><code>
# Clone the repository
git clone https://github.com/yourusername/rag-chatbot-google-docs.git
cd rag-chatbot-google-docs

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
</code></pre>

<h2>🔧 Technologies Used</h2>

<ul>
  <li><strong>Python:</strong> Backend logic</li>
  <li><strong>Streamlit:</strong> Interactive frontend UI</li>
  <li><strong>LangGraph:</strong> Persistent chat flow graph and state management</li>
  <li><strong>LangChain:</strong> Tool integration with LLM (tools, agents, prompts)</li>
  <li><strong>Google APIs:</strong> Docs API & OAuth 2.0 login</li>
  <li><strong>OpenWeather API:</strong> Weather tool</li>
  <li><strong>ExchangeRate API:</strong> Currency converter</li>
  <li><strong>Google Translate API:</strong> Multilingual support</li>
  <li><strong>Wikipedia, YouTube Transcript, PDF extractors:</strong> Custom tools</li>
  <li><strong>SQLite:</strong> Local database for storing chat sessions</li>
</ul>

<h2>📁 Folder Structure</h2>

<pre><code>
.
├── app.py                 # Streamlit frontend
├── backend/               # LLM logic, tools, RAG pipeline
│   ├── chatbot.py         # LangGraph chatbot state engine
│   ├── tools/             # Tool functions: weather, wiki, etc.
│   └── utils.py           # PDF, transcript utilities
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── chatbot.db             # SQLite checkpoint database
</code></pre>

<h2>🌐 Live Demo</h2>

<p><a href="#">👉 Click here to try the live chatbot (Coming Soon)</a></p>

<h2>🤝 Contribution & License</h2>

<p>
Feel free to fork this repo, raise issues or pull requests. All contributions are welcome!<br/><br/>
Licensed under the <strong>MIT License</strong>. See the <code>LICENSE</code> file for details.
</p>
