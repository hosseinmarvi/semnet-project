#! /usr/bin/env python3

from email.message import EmailMessage
import os
import mimetypes
import smtplib
import getpass

message = EmailMessage()
sender = "starrahimi10@gmail.com"
recipient = "setareh10@gmail.com"
message['From'] = sender
message['To'] = recipient
message['Subject'] = '***EMAIL SUBJECT***'
body = """Hi Setareh!
This Email is sent by a python code."""
message.set_content(body)
file_list = []
file_list.append(os.path.expanduser('~') + '/Python/pictures/SD task pictures/SD_all_participants.pdf')
file_list.append(os.path.expanduser('~') + '/Python/pictures/SD task pictures/LD_all_participants.pdf')
for file in file_list:
    attachment_path = file
    attachment_filename = os.path.basename(attachment_path)
    mime_type, _ = mimetypes.guess_type(attachment_path)
    mime_type, mime_subtype = mime_type.split('/', 1)

    with open(attachment_path, 'rb') as ap:
        message.add_attachment(ap.read(),
                               maintype=mime_type,
                               subtype=mime_subtype,
                               filename=attachment_filename)

mail_server = smtplib.SMTP_SSL('smtp.gmail.com')
mail_pass = getpass.getpass('Password? ')   # works on linux terminal
mail_server.login(sender, mail_pass)
mail_server.send_message(message)
mail_server.quit()