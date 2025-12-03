# Web App Guidelines

The hosted version at [https://repuragent.serve.scilifelab.se](https://repuragent.serve.scilifelab.se) delivers Repuragent as a managed service. This chapter explains how to register, what data is stored, and how the cloud deployment protects each user.

## 3.1 Account Registration

1. **Sign up** – click *Register* on UI, enter your email, and choose a strong password.\
 Account information is hashed with Argon2id before being stored in PostgreSQL, so neither admins nor developers ever see plaintext.
2. **Email verification** – an email from repuragent.uu@gmail.com will be sent to your registered email with a one-time verification link. 
3. **Sign in** – once verified, log in to reach the Repuragent Web UI. 
4. **Account recovery** – use *Forgot password* to receive a reset link (valid for 1 h). New passwords are hashed the same way as during sign-up.

If you do not receive any email after registration or a request to change your password, check your spam email folder or contact us via `repuragent.uu@gmail.com`.

The demo thread appears alongside your own conversations; it is a read-only example.

## 3.2 How the Web App Stores and Handles Data

**What is stored permanently?**
- Episodic memory, which is the task decomposition patterns, is stored permanently with the Agent. However, it only stored if you decided to hit the `Extract Learning` button on the web interface. 
- Chat history is also stored, but only for 2 days due to resource limitations. After that, all the conversation and relevant files will be automatically removed. If you want to permanently store conversation history, please consider using the [Local version](local.md).

**What happens if I delete a conversation on the web interface?**
- When you hit Delete conversation on the web interface and confirm, the chat history and all relevant data files, output files will be removed permanently from the database. There is no recovery mechanism to protect your data privacy. 

**What is *not* stored?**
- All your data files and results will not be stored in the system.
- Your data files and output files will be shown to you only and will be removed within 2 days. 
- All this data is not used for training the model.

**Data privacy** 
- Each user has an isolated root directory and a unique thread namespace (UUID). The backend enforces per-user path checks before issuing download tokens.


## 3.3 When to Choose the Web App
- **You need zero installation** - using the web app only requires account registration, then use.
- **You want zero infrastructure** – the hosted service handles Docker, PostgreSQL, ChromaDB, and updates for you.
- **You need multi-user collaboration** – every researcher gets a private workspace.
- **You require centralized data policies** – retention, backups, and access control all live in one deployment.  

