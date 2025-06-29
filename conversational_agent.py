import streamlit as st
import os
import sqlite3
from datetime import datetime
import json
from typing import List, Dict, Optional
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import pandas as pd
from pathlib import Path
import hashlib
import time

# Initialize Ollama components
llm = OllamaLLM(model="mistral")
embeddings = OllamaEmbeddings(model="mistral")

# Text splitter for processing conversations
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)


class ConversationDatabase:
    """SQLite database manager for storing conversation history"""

    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self.conn = None
        self.init_database()

    def init_database(self):
        """Initialize database with required tables"""
        # Use check_same_thread=False for SQLite with Streamlit
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        # Set row factory for dict-like access
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Users table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS users
                       (
                           user_id
                           TEXT
                           PRIMARY
                           KEY,
                           created_at
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           last_active
                           TEXT,
                           metadata
                           TEXT
                       )
                       ''')

        # Conversations table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS conversations
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           user_id
                           TEXT,
                           session_id
                           TEXT,
                           timestamp
                           TEXT
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           user_message
                           TEXT,
                           bot_response
                           TEXT,
                           context_used
                           TEXT,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           user_id
                       )
                           )
                       ''')

        # User preferences table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS user_preferences
                       (
                           user_id
                           TEXT
                           PRIMARY
                           KEY,
                           preferences
                           TEXT,
                           topics_of_interest
                           TEXT,
                           conversation_style
                           TEXT,
                           FOREIGN
                           KEY
                       (
                           user_id
                       ) REFERENCES users
                       (
                           user_id
                       )
                           )
                       ''')

        self.conn.commit()

    def add_user(self, user_id: str, metadata: Dict = None):
        """Add a new user or update last active time"""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        try:
            cursor.execute('''
                           INSERT INTO users (user_id, last_active, metadata)
                           VALUES (?, ?, ?)
                           ''', (user_id, now, json.dumps(metadata or {})))
        except sqlite3.IntegrityError:
            cursor.execute('''
                           UPDATE users
                           SET last_active = ?
                           WHERE user_id = ?
                           ''', (now, user_id))
        self.conn.commit()

    def add_conversation(self, user_id: str, session_id: str, user_message: str,
                         bot_response: str, context_used: str = ""):
        """Store a conversation turn"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute('''
                       INSERT INTO conversations (user_id, session_id, timestamp, user_message, bot_response,
                                                  context_used)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ''', (user_id, session_id, timestamp, user_message, bot_response, context_used))
        self.conn.commit()

    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve user's conversation history"""
        cursor = self.conn.cursor()
        cursor.execute('''
                       SELECT timestamp, user_message, bot_response, context_used
                       FROM conversations
                       WHERE user_id = ?
                       ORDER BY timestamp DESC
                           LIMIT ?
                       ''', (user_id, limit))

        return [dict(row) for row in cursor.fetchall()]

    def get_user_stats(self, user_id: str) -> Dict:
        """Get user statistics"""
        cursor = self.conn.cursor()

        # Total conversations
        cursor.execute('SELECT COUNT(*) as count FROM conversations WHERE user_id = ?', (user_id,))
        total_conversations = cursor.fetchone()['count']

        # First and last interaction
        cursor.execute('''
                       SELECT MIN(timestamp) as first, MAX(timestamp) as last
                       FROM conversations
                       WHERE user_id = ?
                       ''', (user_id,))
        result = cursor.fetchone()
        first_interaction = result['first'] if result else None
        last_interaction = result['last'] if result else None

        # Get all messages for topic extraction
        cursor.execute('SELECT user_message FROM conversations WHERE user_id = ?', (user_id,))
        messages = [row['user_message'] for row in cursor.fetchall()]

        return {
            'total_conversations': total_conversations,
            'first_interaction': first_interaction,
            'last_interaction': last_interaction,
            'messages': messages
        }

    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                           INSERT INTO user_preferences (user_id, preferences, topics_of_interest, conversation_style)
                           VALUES (?, ?, ?, ?)
                           ''', (
                               user_id,
                               json.dumps(preferences.get('preferences', {})),
                               json.dumps(preferences.get('topics', [])),
                               preferences.get('style', 'casual')
                           ))
        except sqlite3.IntegrityError:
            cursor.execute('''
                           UPDATE user_preferences
                           SET preferences        = ?,
                               topics_of_interest = ?,
                               conversation_style = ?
                           WHERE user_id = ?
                           ''', (
                               json.dumps(preferences.get('preferences', {})),
                               json.dumps(preferences.get('topics', [])),
                               preferences.get('style', 'casual'),
                               user_id
                           ))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class ConversationalAgent:
    """Main conversational agent with memory and context awareness"""

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.db = ConversationDatabase()
        self.vector_store_path = f"vector_stores/{user_id}"
        self.vector_store = None

        # Create directories
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)

        # Initialize user in database
        self.db.add_user(user_id)

        # Load or create vector store
        self.init_vector_store()

        # Conversation prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful, friendly conversational assistant with access to the user's conversation history. 
            Use the following context from past conversations to provide personalized and contextually aware responses:

            User Profile:
            - User ID: {user_id}
            - Total past conversations: {total_conversations}
            - Recent topics discussed: {recent_topics}

            Relevant Context from Past Conversations:
            {context}

            Guidelines:
            - Be conversational and natural
            - Reference past conversations when relevant
            - Remember user preferences and interests
            - Ask follow-up questions to keep the conversation engaging
            - Be helpful while maintaining a friendly tone
            - If you notice patterns in user interests, acknowledge them"""),
            ("human", "{input}")
        ])

    def init_vector_store(self):
        """Initialize or load vector store for semantic search"""
        vector_store_file = os.path.join(self.vector_store_path, "index.faiss")

        try:
            if os.path.exists(vector_store_file):
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    embeddings,
                    index_name="index",
                    allow_dangerous_deserialization=True
                )
            else:
                # Create initial vector store
                initial_doc = Document(
                    page_content=f"Conversation history for user {self.user_id} initialized",
                    metadata={"timestamp": datetime.now().isoformat(), "type": "system"}
                )
                self.vector_store = FAISS.from_documents([initial_doc], embeddings)
                self.save_vector_store()
        except Exception as e:
            st.error(f"Error with vector store: {e}")
            # Fallback to new vector store
            initial_doc = Document(
                page_content=f"Conversation history for user {self.user_id} initialized",
                metadata={"timestamp": datetime.now().isoformat(), "type": "system"}
            )
            self.vector_store = FAISS.from_documents([initial_doc], embeddings)

    def save_vector_store(self):
        """Save vector store to disk"""
        try:
            self.vector_store.save_local(self.vector_store_path, index_name="index")
        except Exception as e:
            st.error(f"Error saving vector store: {e}")

    def add_to_memory(self, user_input: str, bot_response: str, context_used: str = ""):
        """Add conversation to both database and vector store"""
        timestamp = datetime.now().isoformat()

        # Add to database
        self.db.add_conversation(
            self.user_id,
            self.session_id,
            user_input,
            bot_response,
            context_used
        )

        # Create documents for vector store
        conversation_doc = Document(
            page_content=f"User: {user_input}\nAssistant: {bot_response}",
            metadata={
                "timestamp": timestamp,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "type": "conversation"
            }
        )

        # Add to vector store
        self.vector_store.add_documents([conversation_doc])
        self.save_vector_store()

    def search_relevant_context(self, query: str, k: int = 3) -> str:
        """Search for relevant context from past conversations"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            context_parts = []
            for doc in docs:
                if doc.metadata.get('type') != 'system':
                    context_parts.append(doc.page_content)
            return "\n\n".join(context_parts)
        except Exception as e:
            st.error(f"Error searching context: {e}")
            return ""

    def extract_topics_from_history(self) -> List[str]:
        """Extract main topics from conversation history"""
        stats = self.db.get_user_stats(self.user_id)
        messages = stats.get('messages', [])

        if not messages:
            return []

        # Simple topic extraction (you could use more sophisticated NLP here)
        all_text = " ".join(messages).lower()
        words = all_text.split()

        # Filter common words and get frequent terms
        common_words = set(
            ['the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as',
             'by', 'that', 'this', 'it', 'from', 'be', 'are', 'was', 'were', 'been'])
        meaningful_words = [w for w in words if len(w) > 3 and w not in common_words]

        # Get top 5 most common topics
        from collections import Counter
        word_counts = Counter(meaningful_words)
        return [word for word, count in word_counts.most_common(5)]

    def generate_response(self, user_input: str) -> str:
        """Generate a contextual response"""
        # Search for relevant context
        context = self.search_relevant_context(user_input)

        # Get user stats
        stats = self.db.get_user_stats(self.user_id)
        recent_topics = self.extract_topics_from_history()

        # Generate response
        try:
            response = self.prompt | llm
            result = response.invoke({
                "user_id": self.user_id,
                "total_conversations": stats['total_conversations'],
                "recent_topics": ", ".join(recent_topics) if recent_topics else "None yet",
                "context": context if context else "No previous relevant conversations found",
                "input": user_input
            })

            # Extract response text
            if hasattr(result, 'content'):
                response_text = result.content
            else:
                response_text = str(result)

            # Store in memory
            self.add_to_memory(user_input, response_text, context[:200] if context else "")

            return response_text

        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error. Could you please try again?"


# Streamlit UI with professional design
def main():
    st.set_page_config(
        page_title="AI Assistant with Memory",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Professional CSS styling
    st.markdown("""
        <style>
        /* Main app styling */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* Chat container styling */
        .chat-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        /* Message styling */
        .stChatMessage {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        /* User message specific */
        .stChatMessage[data-testid="user-message"] {
            background-color: #007bff;
            color: white;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2c3e50;
            color: white;
        }

        /* Metrics styling */
        [data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        /* Button styling */
        .stButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s;
        }

        .stButton > button:hover {
            background-color: #0056b3;
            transform: translateY(-1px);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }

        /* Topic badges */
        .topic-badge {
            display: inline-block;
            background-color: #007bff;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            margin: 3px;
            font-size: 14px;
            font-weight: 500;
        }

        /* Header styling */
        h1 {
            color: #2c3e50;
            font-weight: 700;
            text-align: center;
            padding: 20px 0;
        }

        h2, h3 {
            color: #34495e;
        }

        /* Info box styling */
        .info-box {
            background-color: #e8f4fd;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with gradient
    st.markdown("""
        <h1 style='text-align: center; color: #2c3e50;'>
            🧠 AI Assistant with Persistent Memory
        </h1>
        <p style='text-align: center; color: #7f8c8d; font-size: 18px;'>
            I remember our conversations and learn about you over time!
        </p>
    """, unsafe_allow_html=True)

    # Sidebar with professional styling
    with st.sidebar:
        st.markdown("## 👤 User Profile")

        # User ID input with better styling
        user_id = st.text_input(
            "User ID",
            value="default_user",
            key="user_id",
            help="Enter a unique ID to maintain separate conversation histories"
        )

        # Initialize agent
        if 'agent' not in st.session_state or st.session_state.get('current_user') != user_id:
            with st.spinner("Loading your profile..."):
                st.session_state.agent = ConversationalAgent(user_id)
                st.session_state.current_user = user_id
                st.session_state.messages = []

        agent = st.session_state.agent

        # User statistics with better visualization
        st.markdown("## 📊 Your Statistics")
        stats = agent.db.get_user_stats(user_id)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Total Chats",
                f"{stats['total_conversations']:,}",
                delta="Active" if stats['total_conversations'] > 0 else None
            )
        with col2:
            if stats['last_interaction']:
                last_chat = datetime.fromisoformat(stats['last_interaction'])
                time_diff = datetime.now() - last_chat
                if time_diff.days == 0:
                    time_str = "Today"
                elif time_diff.days == 1:
                    time_str = "Yesterday"
                else:
                    time_str = f"{time_diff.days}d ago"
                st.metric("Last Chat", time_str)
            else:
                st.metric("Last Chat", "Never")

        # Topics of interest with badges
        topics = agent.extract_topics_from_history()
        if topics:
            st.markdown("## 🏷️ Your Topics")
            topics_html = "".join([f'<span class="topic-badge">{topic}</span>' for topic in topics])
            st.markdown(topics_html, unsafe_allow_html=True)

        # Conversation history with better formatting
        st.markdown("## 📜 Recent History")
        history = agent.db.get_user_history(user_id, limit=5)

        if history:
            for i, conv in enumerate(history):
                timestamp = datetime.fromisoformat(conv['timestamp'])
                with st.expander(f"💬 {timestamp.strftime('%b %d, %I:%M %p')}"):
                    st.markdown(f"**You:** {conv['user_message']}")
                    st.markdown(f"**Assistant:** {conv['bot_response']}")
        else:
            st.info("No conversation history yet. Start chatting!")

        # Management section
        st.markdown("## ⚙️ Options")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Session", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("🗑️ Reset All Data", use_container_width=True):
                if st.checkbox("I'm sure", key="confirm_reset"):
                    # Clear vector store
                    import shutil
                    if os.path.exists(agent.vector_store_path):
                        shutil.rmtree(agent.vector_store_path)

                    # Clear database entries
                    cursor = agent.db.conn.cursor()
                    cursor.execute('DELETE FROM conversations WHERE user_id = ?', (user_id,))
                    cursor.execute('DELETE FROM user_preferences WHERE user_id = ?', (user_id,))
                    agent.db.conn.commit()

                    # Reinitialize
                    st.session_state.agent = ConversationalAgent(user_id)
                    st.session_state.messages = []
                    st.success("All your data has been reset!")
                    time.sleep(1)
                    st.rerun()

        # Export section
        st.markdown("## 📤 Export Data")
        if st.button("📥 Download History", use_container_width=True):
            history_df = pd.DataFrame(agent.db.get_user_history(user_id, limit=1000))
            if not history_df.empty:
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{user_id}_conversations_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No data to export")

    # Main chat area
    chat_col, info_col = st.columns([2, 1])

    with chat_col:
        # Chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Type your message here...", key="chat_input"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    response = agent.generate_response(prompt)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with info_col:
        # Quick stats card
        st.markdown("""
            <div class="info-box">
                <h3>💡 Quick Tips</h3>
                <ul>
                    <li>I remember our past conversations</li>
                    <li>Ask me about topics we've discussed</li>
                    <li>Your data is stored locally</li>
                    <li>Use different User IDs for separate profiles</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # How it works section
        with st.expander("ℹ️ How it works", expanded=False):
            st.markdown("""
            ### Features:
            - **🧠 Persistent Memory**: All conversations stored in SQLite
            - **🔍 Semantic Search**: Vector embeddings for context
            - **👤 User Profiles**: Separate histories per user
            - **📊 Analytics**: Track conversation patterns
            - **🔒 Privacy**: Local data storage

            ### Architecture:
            - **LLM**: Ollama Mistral for responses
            - **Database**: SQLite for conversation history
            - **Vector Store**: FAISS for semantic search
            - **Framework**: Streamlit for UI
            """)

        # System status
        with st.expander("🔧 System Status", expanded=False):
            st.success("✅ Database: Connected")
            st.success("✅ Vector Store: Loaded")
            st.success("✅ LLM: Ready")

            # Show storage info
            if os.path.exists(agent.vector_store_path):
                size = sum(os.path.getsize(os.path.join(agent.vector_store_path, f))
                           for f in os.listdir(agent.vector_store_path))
                st.info(f"💾 Storage Used: {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()