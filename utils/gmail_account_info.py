# gmail token to allow python to access gmail for sending emails
# first, set up 2-factor authorization on myaccount.google.com
# then, set up "app passwords" and get a token for using mail on a Windows computer,
# and use this token in place of the account password when using the email code in email_utils.py
sender_email = 'YOUR_DESIRED_ZOMBIE_ALERTER_EMAIL_HERE@gmail.com'
gmail_token = 'YOUR_2FA_TOKEN_HERE'