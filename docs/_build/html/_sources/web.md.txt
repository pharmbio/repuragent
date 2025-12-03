# Web App Guidelines

The hosted version at [https://repuragent.serve.scilifelab.se](https://repuragent.serve.scilifelab.se)
delivers Repuragent as a managed service. This chapter explains how to register, what data
is stored, and how the cloud deployment protects each user.

## 3.1 Account Registration

1. **Sign up**  
   Click *Register* on the UI, enter your email, and choose a strong password.
   Account information is hashed with Argon2id before being stored in PostgreSQL, so
   neither admins nor developers ever see plaintext.
2. **Email verification**  
   An email from [repuragent.uu@gmail.com](mailto:repuragent.uu@gmail.com) will be sent to your registered email with a one-time verification link.
3. **Sign in**  
   Once verified, log in to reach the Repuragent Web UI.
4. **Account recovery**  
   Use *Forgot password* to receive a reset link (valid for 1 h). New passwords are hashed the same way as during sign-up.

If you do not receive any email after registration or a request to change your password, check your spam email folder or contact us via [repuragent.uu@gmail.com](mailto:repuragent.uu@gmail.com).

The demo thread appears alongside your own conversations; it is a read-only example.

## 3.2 How the Web App Stores and Handles Data

- **What is stored permanently?**
  - Episodic memory (task decomposition patterns) is stored permanently with the Agent,
    but only if you hit the `Extract Learning` button on the web interface.
  - Chat history is also stored, but only for 2 days due to resource limitations. After
    that, all conversations and relevant files are automatically removed. For permanent
    history, consider the [Local version](local.md).

- **What happens if I delete a conversation on the web interface?**
  - When you delete a conversation and confirm, the chat history and all related data/output
    files are removed permanently from the database. There is no recovery mechanism.

- **What is *not* stored?**
  - Data files and results are not retained in the system.
  - Your files and outputs are only visible to you and are removed within 2 days.
  - None of this data is used for training the model.

- **Data privacy**
  - Each user has an isolated root directory and a unique thread namespace (UUID). The
    backend enforces per-user path checks before issuing download tokens.


## 3.3 When to Choose the Web App
- **You need zero installation**  
  Using the web app only requires account registration, then use.
- **You want zero infrastructure**  
  The hosted service handles Docker, PostgreSQL, ChromaDB, and updates for you.
- **You need multi-user collaboration**  
  Every researcher gets a private workspace.
- **You require centralized data policies**  
  Retention, backups, and access control all live in one deployment.
