# Web App Guidelines

The hosted version at [https://repuragent.serve.scilifelab.se](https://repuragent.serve.scilifelab.se)
delivers Repuragent as a web application. This chapter explains how to register an account, what data
is stored, and how the cloud deployment protects each user.

## 3.1 Account Registration

1. **Sign up**  
   Click *Register* on the UI, enter your email, and choose a strong password.
   Account information is hashed before being stored in PostgreSQL.
2. **Email verification**  
   An email from [repuragent.uu@gmail.com](mailto:repuragent.uu@gmail.com) will be sent to your registered email with a one-time verification link.
3. **Sign in**  
   Once verified, log in to access Repuragent's full functionality. The demo conversation appears once you log in; it is a read-only example. Create a new task to start your own conversations.
4. **Account recovery**  
   Use *Forgot password* to receive a reset link.

If you do not receive any email after registration or a request to change your password, check your spam email folder or contact us via [repuragent.uu@gmail.com](mailto:repuragent.uu@gmail.com).

## 3.2 How the Repuragent stores and handles data

- **What is stored in the system?**
  - When you hit hit the *Extract Learning* button on the web interface, the Repuragent will analyse your current conversation and store the task decomposition patterns permanently into Episodic memory. Note that, it jsut happen when you hit *Extract Learning* button. 
  - Chat history is also stored, but only for 2 days due to resource limitations. After
  that, all conversations and relevant files are automatically removed. For permanent
  history, consider the [Local version](local.md).

- **What happens if I delete a conversation on the web interface?**
  - When you delete a conversation and confirm, the chat history and all related data/output
    files are removed permanently from the database. There is no recovery mechanism.

- **What is *not* stored?**
  - None of your data is used for training the model.
  - Data files and results are not exposed to any other users.
  - Your files and outputs are only visible to you and are removed within 2 days.