import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
from datetime import datetime
import os
import webbrowser
import re
import json
from ai_collabration import ChatbotConfig,DocumentChatbot,DocumentLoader
import tempfile
from git import Repo

class ChatbotInterface:
    """
    Main class for the Aarvya AI Assistant interface.
    Handles all UI components and chat functionality.
    """

    def __init__(self, root):
        """
        Initialize the chat interface.
        Args:
            root: The main window (ttk.Window instance)
        """
        self.ch = None
        self.root = root
        self.setup_window()
        self.setup_themes()
        self.setup_history_storage()
        self.setup_ui()
        self.bind_events()
        self.config = ChatbotConfig()
        self.chatbot = DocumentChatbot(self.config)
        # Constants
        self.THEME_ICONS = {"dark": "☀️", "light": "🌙"}
        self.FILE_TYPES = {
            "pdf": [("PDF files", "*.pdf")],
            "audio": [("Audio files", "*.mp3 *.wav")],
            "video": [("Video files", "*.mp4 *.avi *.mkv")]
        }

    def setup_window(self):
        """Configure main window properties and geometry"""
        self.root.title("Aarvya AI Assistant")
        self.root.geometry("1200x800")
        
    def setup_themes(self):
        """
        Initialize theme settings and color schemes for both dark and light modes.
        Configures custom styles for different UI components.
        """
        self.is_dark_theme = True
        self.style = ttk.Style()
        self.style.theme_use('darkly')
        
        # Theme colors
        self.colors = {
            'dark': {
                'bg': '#1a1a1a',
                'content_bg': '#2d2d2d',
                'text': '#ffffff',
                'text_secondary': '#a0a0a0'
            },
            'light': {
                'bg': '#ffffff',
                'content_bg': '#f5f5f5',
                'text': '#000000',
                'text_secondary': '#666666'
            }
        }
        
        # Apply initial theme
        self._apply_theme_styles()

    def _apply_theme_styles(self):
        """Helper method to apply theme styles to all components"""
        current = 'dark' if self.is_dark_theme else 'light'
        colors = self.colors[current]
        
        # Configure styles
        self.style.configure('Sidebar.TFrame', background=colors['bg'])
        self.style.configure('Content.TFrame', background=colors['content_bg'])
        self.style.configure('Title.TLabel', 
                           font=('Segoe UI', 24, 'bold'),
                           foreground=colors['text'],
                           background=colors['content_bg'])

    def setup_ui(self):
        """Setup UI components"""
        self.setup_main_frame()
        self.setup_sidebar()
        self.setup_content_area()

    def setup_main_frame(self):
        """Setup main container frame"""
        self.main_frame = ttk.Frame(self.root, style='Content.TFrame')
        self.main_frame.pack(fill=BOTH, expand=YES)

    def setup_sidebar(self):
        """Setup sidebar with enhanced styling"""
        self.sidebar = ttk.Frame(self.main_frame, style='Sidebar.TFrame', width=300)
        self.sidebar.pack(side=LEFT, fill=Y, padx=0)
        self.sidebar.pack_propagate(False)
        
        # Logo and branding
        brand_frame = ttk.Frame(self.sidebar, style='Sidebar.TFrame')
        brand_frame.pack(fill=X, pady=20, padx=15)
        
        ttk.Label(
            brand_frame,
            text="🤖 Aarvya AI",
            font=("Segoe UI", 20, "bold"),
            foreground='#ffffff',
            background='#1a1a1a'
        ).pack(side=LEFT)
        
        # Separator
        ttk.Separator(self.sidebar, bootstyle="light").pack(fill=X, padx=15)
        
        # Action buttons with improved styling
        buttons_frame = ttk.Frame(self.sidebar, style='Sidebar.TFrame')
        buttons_frame.pack(fill=X, pady=20, padx=15)
        
        button_configs = [
            ("✨ New Chat", "success-outline", self.new_chat),
            ("📜 History", "warning-outline", self.show_history),  # Add this line
            ("💾 Save Chat", "info-outline", self.save_chat),
            ("🗑️ Clear Chat", "danger-outline", self.clear_chat)
        ]
        
        for text, style, command in button_configs:
            btn = ttk.Button(
                buttons_frame,
                text=text,
                bootstyle=style,
                command=command,
                width=25,
                padding=10
            )
            btn.pack(pady=5, fill=X)
        
        # File handling section
        ttk.Label(
            self.sidebar,
            text="File Operations",
            font=("Segoe UI", 14, "bold"),
            foreground='#ffffff',
            background='#1a1a1a'
        ).pack(pady=(20, 10), padx=15)
        
        file_buttons = [
            ("📁 Open Folder", self.open_folder),
            ("📄 Open PDF", self.open_pdf),
            ("🎵 Open Audio", self.open_audio),
            ("🎬 Open Video", self.open_video),
        ]
        
        file_frame = ttk.Frame(self.sidebar, style='Sidebar.TFrame')
        file_frame.pack(fill=X, padx=15)
        
        for text, command in file_buttons:
            ttk.Button(
                file_frame,
                text=text,
                bootstyle="secondary-link",
                command=command,
                width=25,
                padding=5
            ).pack(pady=2, fill=X)

    def setup_content_area(self):
        """Setup main content area with improved styling"""
        content_frame = ttk.Frame(self.main_frame, style='Content.TFrame')
        content_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(content_frame, style='Content.TFrame')
        header_frame.pack(fill=X, pady=(0, 20))
        
        ttk.Label(
            header_frame,
            text="Chat Assistant",
            style='Title.TLabel'
        ).pack(side=LEFT)
        
        self.theme_btn = ttk.Button(
            header_frame,
            text="🌙" if not self.is_dark_theme else "☀️",
            command=self.toggle_theme,
            bootstyle="link-outline",
            width=3
        )
        self.theme_btn.pack(side=RIGHT)
        
        # Chat display
        self.chat_display = ScrolledText(
            content_frame,
            padding=15,
            height=20,
            autohide=True,
            bootstyle="round"
        )
        self.chat_display.pack(fill=BOTH, expand=YES, pady=(0, 20))
        
        # Input area
        input_frame = ttk.Frame(content_frame, style='Content.TFrame')
        input_frame.pack(fill=X, side=BOTTOM)
        
        self.prompt_input = ScrolledText(
            input_frame,
            height=4,
            padding=10,
            autohide=True,
            bootstyle="round"
        )
        self.prompt_input.pack(side=LEFT, fill=X, expand=YES, padx=(0, 10))
        
        self.submit_btn = ttk.Button(
            input_frame,
            text="Send ➤",
            command=self.submit_prompt,
            bootstyle="primary",
            width=10,
            padding=10
        )
        self.submit_btn.pack(side=RIGHT)

    def bind_events(self):
        """Bind keyboard shortcuts"""
        self.prompt_input.bind('<Control-Return>', lambda e: self.submit_prompt())

    def submit_prompt(self):
        """
        Handle user prompt submission.
        Processes user input, displays it in chat, and generates AI response.
        Saves chat history after each interaction.
        """
        if self.ch == 1:
            prompt = self.prompt_input.get('1.0', 'end-1c').strip()
            if prompt:
                answer = (self.chatbot).query(prompt)
                timestamp = self.get_timestamp()
                self.save_history()
        
        # Add messages to chat
                self._add_message("You", prompt, "user_message")
                self._add_message("AI Assistant", f"Thank you for your message: {answer}", "bot_message")
        
        # Clear input and save
                self.prompt_input.delete('1.0', tk.END)
                self.save_history()


        prompt = self.prompt_input.get('1.0', 'end-1c').strip()
        if not prompt:
            return
        
        question = prompt
        
        if question:
            answer = (self.chatbot).query_directly_to_llm(question)
            timestamp = self.get_timestamp()
        
        # Add messages to chat
            self._add_message("You", prompt, "user_message")
            self._add_message("AI Assistant", f"Thank you for your message: {answer}", "bot_message")
        
        # Clear input and save
            self.prompt_input.delete('1.0', tk.END)
            self.save_history()

    def _add_message(self, sender, content, tag):
        """
        Helper method to add messages to chat display.
        Args:
            sender (str): Message sender
            content (str): Message content
            tag (str): Style tag for the message
        """
        timestamp = self.get_timestamp()
        self.chat_display.insert(tk.END, f"\n{timestamp} {sender}:\n", "timestamp")
        self.chat_display.insert(tk.END, f"{content}\n", tag)
        self.chat_display.see(tk.END)

    def get_timestamp(self):
        return datetime.now().strftime("[%H:%M:%S]")

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self.theme_btn.config(text="☀️" if self.is_dark_theme else "🌙")
        
        if self.is_dark_theme:
            self.style.theme_use('darkly')
        else:
            self.style.theme_use('litera')

    # Added missing methods
    def new_chat(self):
        if messagebox.askyesno("New Chat", "Start a new chat? Current chat will be cleared."):
            self.chat_display.delete('1.0', tk.END)
            self.prompt_input.delete('1.0', tk.END)
            welcome_msg = "Start a new conversation! How can I help you today?\n"
            self.chat_display.insert(tk.END, welcome_msg, "bot_message")
            self.chat_display.see(tk.END)

    def save_chat(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", ".txt"), ("All files", ".*")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(self.chat_display.get('1.0', tk.END))
            messagebox.showinfo("Success", "Chat saved successfully!")

    def clear_chat(self):
        if messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat?"):
            self.chat_display.delete('1.0', tk.END)
            self.ch = None

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self._add_message("You", f"Selected PDF: {folder_path}", "user_message")
            docs = self.chatbot.load_directory(folder_path)
            self.chatbot.initialize_qa_chain(docs)
            self._add_message("AI Assistant", "directory loaded successfully!", "bot_message")
            self.ch = 1


    def open_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        self._add_message("You", f"Selected PDF: {file_path}", "user_message")
        docs = DocumentLoader.load_file(file_path, self.config)
        self.chatbot.initialize_qa_chain(docs)
        self._add_message("AI Assistant", "PDF loaded successfully!", "bot_message")
        self.ch = 1
        
        

    def open_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.mp3 *.wav")])
        if file_path:
            self.prompt_input.insert(tk.END, f"\nSelected audio: {file_path}")

    def open_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.prompt_input.insert(tk.END, f"\nSelected video: {file_path}")

    def create_repo_dialog(self):
        """Create a dialog window for GitHub repository URL input"""
        dialog = tk.Toplevel(self.root)
        dialog.title("GitHub Repository")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + self.root.winfo_width()/2 - 200,
            self.root.winfo_rooty() + self.root.winfo_height()/2 - 75
        ))
        
        # Message
        ttk.Label(
            dialog,
            text="Enter GitHub Repository URL:",
            font=("Segoe UI", 11)
        ).pack(pady=(20, 10))
        
        # URL entry
        url_entry = ttk.Entry(dialog, width=50)
        url_entry.pack(pady=10, padx=20)
        url_entry.focus()
        
        def submit_url():
            repo_url = url_entry.get().strip()
            github_pattern = r"^https?://github\.com/[\w-]+/[\w-]+/?$"
            
            if repo:
                timestamp = self.get_timestamp()
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo = Repo.clone_from(repo_url, temp_dir)
                    if repo:
                        docs = chatbot.load_directory(repo.working_tree_dir)
                        self.chatbot.initialize_qa_chain(docs)
                        self._add_message("AI Assistant", "github repo loaded successfully!", "bot_message")
                        self.ch = 1
                
                # Add to chat display
                self.chat_display.insert(tk.END, f"\n{timestamp} You:\n", "timestamp")
                self.chat_display.insert(tk.END, f"Opening GitHub Repository: {repo_url}\n", "user_message")
                
                # Open in browser
                webbrowser.open(repo_url)
                
                # Bot response
                self.chat_display.insert(tk.END, f"\n{timestamp} AI Assistant:\n", "timestamp")
                self.chat_display.insert(tk.END, f"Repository opened in your browser.\n", "bot_message")
                self.chat_display.see(tk.END)
                dialog.destroy()
            else:
                messagebox.showerror("Invalid URL", 
                    "Please enter a valid GitHub repository URL\nExample: https://github.com/username/repository",
                    parent=dialog)

        # Submit button
        submit_btn = ttk.Button(
            dialog,
            text="Submit",
            command=submit_url,
            style="primary.TButton",
            width=20
        )
        submit_btn.pack(pady=10)
        
        # Bind Enter key to submit
        url_entry.bind('<Return>', lambda e: submit_url())

    def github_repo(self):
        """Open GitHub repository dialog"""
        self.create_repo_dialog()

    def setup_history_storage(self):
        """Initialize history storage"""
        self.history_file = "chat_history.json"
        self.load_history()

    def load_history(self):
        """Load chat history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as file:
                    self.chat_history = json.load(file)
            else:
                self.chat_history = []
        except Exception:
            self.chat_history = []

    def save_history(self):
        """
        Save chat history to JSON file.
        Includes timestamp and content for each chat session.
        """
        try:
            chat_content = self.chat_display.get('1.0', tk.END).strip()
            if not chat_content:
                return
                
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = {'timestamp': timestamp, 'content': chat_content}
            
            self.chat_history.append(entry)
            
            with open(self.history_file, 'w', encoding='utf-8') as file:
                json.dump(self.chat_history, file, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save history: {str(e)}")

    def show_history(self):
        """Show history dialog"""
        history_dialog = tk.Toplevel(self.root)
        history_dialog.title("Chat History")
        history_dialog.geometry("800x400")
        history_dialog.transient(self.root)
        history_dialog.grab_set()
        
        # Center the dialog
        history_dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + self.root.winfo_width()/2 - 300,
            self.root.winfo_rooty() + self.root.winfo_height()/2 - 200
        ))
        
        # History list
        history_frame = ttk.Frame(history_dialog)
        history_frame.pack(fill=BOTH, expand=YES, padx=20, pady=20)
        
        # Scrollable history list
        history_list = ScrolledText(
            history_frame,
            height=15,
            padding=10,
            autohide=True,
            bootstyle="round"
        )
        history_list.pack(fill=BOTH, expand=YES)
        
        # Populate history
        for entry in self.chat_history:
            timestamp = entry['timestamp']
            content = entry['content']
            history_list.insert(tk.END, f"=== {timestamp} ===\n", "timestamp")
            history_list.insert(tk.END, f"{content}\n\n", "content")
        
        # Style tags
        history_list.tag_configure("timestamp", foreground="#a0a0a0", font=("Segoe UI", 9, "bold"))
        history_list.tag_configure("content", font=("Segoe UI", 10))
        
        # Buttons frame
        button_frame = ttk.Frame(history_dialog)
        button_frame.pack(fill=X, padx=20, pady=(0, 20))
        
        # Load selected chat button
        load_btn = ttk.Button(
            button_frame,
            text="Load Selected",
            command=lambda: self.load_selected_chat(history_list),
            bootstyle="success-outline",
            width=15
        )
        load_btn.pack(side=LEFT, padx=5)
        
        # Clear history button
        clear_btn = ttk.Button(
            button_frame,
            text="Clear History",
            command=lambda: self.clear_history(history_dialog),
            bootstyle="danger-outline",
            width=15
        )
        clear_btn.pack(side=RIGHT, padx=5)

    def load_selected_chat(self, history_list):
        """Load selected chat into main window"""
        try:
            selected = history_list.get("sel.first", "sel.last")
            if selected:
                if messagebox.askyesno("Load Chat", "Load selected chat? Current chat will be cleared."):
                    self.chat_display.delete('1.0', tk.END)
                    self.chat_display.insert(tk.END, selected)
                    self.chat_display.see(tk.END)
        except tk.TclError:
            messagebox.showwarning("Selection", "Please select a chat to load")

    def clear_history(self, dialog):
        """Clear all chat history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all chat history?"):
            self.chat_history = []
            try:
                os.remove(self.history_file)
            except Exception:
                pass
            dialog.destroy()

def main():
    root = ttk.Window(themename="darkly")
    app = ChatbotInterface(root)
    root.mainloop()

if __name__ == "__main__":
    main()