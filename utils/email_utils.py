# -*- coding: utf-8 -*-
"""
Code for sending email alerts for summiting classifier
Can send to multiple recipients, add attachments, etc. 

References:
    https://realpython.com/python-send-email/
    https://stackoverflow.com/questions/23171140/how-do-i-send-an-email-with-a-csv-attachment-using-python
"""

import socket
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
import ssl
import os
import sys

file_path = os.path.abspath(__file__)
utils_dir = os.path.abspath(os.path.join(file_path, os.pardir))
sys.path.append(utils_dir)

def is_there_internet(host="8.8.8.8", port=53, timeout=3):
    '''
    Checks if internet connection is available. 
    If there isn't and you try to send an email, code will crash.
    Reference: https://stackoverflow.com/questions/3764291/checking-network-connection
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    '''
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False
    
def email(recipient, subject, content, filesToSend=[]):
    '''
    Taken from here:
    https://stackoverflow.com/questions/25346001/add-excel-file-attachment-when-sending-python-email
    '''
    if type(recipient) != list:
        recipient = [recipient]
        
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = COMMASPACE.join(recipient)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach( MIMEText(content) )
    
    filesToSend = [os.path.abspath(f) for f in filesToSend if os.path.exists(f)]
    
    context = ssl.create_default_context()

    for file in filesToSend:
        part = MIMEBase('application', "octet-stream")
        part.set_payload( open(file,"rb").read() )
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"'
                       % os.path.basename(file))
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    #server.set_debuglevel(1)
    server.ehlo_or_helo_if_needed()
    server.starttls(context=context)
    server.ehlo_or_helo_if_needed()
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, recipient, msg.as_string())
    server.quit()


def send_email(recipient, subject='subject', content='content', filesToSend=[]):
    ''' 
    Wrapper for email()
    '''
    success = 1; status = 'email sent'
    
    # check if there's internet
    if not is_there_internet():
        success = 0; status = 'No Internet'
    
    # try sending an email
    try:
        assert type(recipient)==list
        assert type(filesToSend)==list
        email(recipient, subject, content, filesToSend)
    
    # if doesn't work, return why
    except Exception as e:
        print(e)
        # try without adding attachments
        try:
            if len(filesToSend) > 0:
                new_header = "~!"*30 + "\nCOULDN'T ATTACH FILES:\n"
                for fil in filesToSend:
                    new_header += f'\t{fil}\n'
                content = content + '\n' + new_header + "~!"*30 + '\n'
            email(recipient, subject, content, filesToSend=[])
        except Exception as e2:
            success = 0; status = e2      
        
    return success, status

#### SPECIFY THE GMAIL ADDDRESS (sender_email)
#### AND THE GMAIL TOKEN FROM SETTING UP 2FA ON myaccount.google.com
#### IN gmail_account_info.py IN THIS DIRECTORY (utils/)
from gmail_account_info import gmail_token as sender_password, sender_email
