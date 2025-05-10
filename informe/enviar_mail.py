import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


EMAIL_REMITENTE = 'beniciomanzotti@gmail.com'
EMAIL_DESTINATARIO = 'beniciomanzotti@gmail.com'
CLAVE_APP = 'gxtjebgpvaadhkgz'


def enviar_email(asunto, cuerpo):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_REMITENTE
    msg['To'] = EMAIL_DESTINATARIO
    msg['Subject'] = asunto
    msg.attach(MIMEText(cuerpo, 'html'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_REMITENTE, CLAVE_APP)
        server.sendmail(EMAIL_REMITENTE, EMAIL_DESTINATARIO, msg.as_string())
        server.quit()
        print("✅ Email enviado con éxito")
    except Exception as e:
        print(f"❌ Error al enviar email: {e}")
