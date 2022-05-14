import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

def send_email(subject="No subject", content="I am boring"):
  mail_host = "smtp.163.com"       # 邮箱的设置里的服务器地址/Server address.Find it in mailbox settings
  mail_user = "*****@163.com"      # 发送邮件的邮箱/Email address for sending mail
  mail_pw = "*********"            # 授权码，邮箱设置里开启POP3/SMTP服务，提供给你的密钥/Authorization code, the key provided to you by opening POP3 / SMTP service in mailbox settings
  sender = "******@163.com"        # 发送邮件的邮箱/Email address for sending mail
  receiver = "******@icloud.com"   # 接收邮件的邮箱/Email address for receiving mail

  # Create the container (outer) email message.
  msg = MIMEText(content, "plain", "utf-8")
  msg['Subject'] = subject
  msg['From'] = sender
  msg['To'] = receiver

  try:
    smtp = smtplib.SMTP_SSL(mail_host, 994)   # 实例化smtp服务器
    smtp.login(mail_user, mail_pw)            # 登录
    smtp.sendmail(sender, receiver, msg.as_string())
    print("Email send successfully")
  except smtplib.SMTPException:
    print("Error: email send failed")

if __name__ == '__main__':
  send_email(subject="Training finished", content="Test")
