import smtplib
# from email.mime.text import MIMEText
#
# # Open a plain text file for reading.  For this example, assume that
# # the text file contains only ASCII characters.
# fp = open(textfile, 'rb')
# # Create a text/plain message
# msg = MIMEText(fp.read())
# fp.close()

# me == the sender's email address
# you == the recipient's email address
content = "test content..."
me = "unswtalk.assignment@gmail.com"
you = "jamison.tsai@gmail.com"
print("test")
# Send the message via our own SMTP server, but don't include the
# envelope header.
s = smtplib.SMTP('smtp.gmail.com',587)
s.starttls()
s.login("test.auto.email.list@gmail.com","testabc123")
s.sendmail(me, you, content)
s.quit()
