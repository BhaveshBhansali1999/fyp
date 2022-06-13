from twilio.rest import Client
def send_message(name):
    # Your Account SID from twilio.com/console
    account_sid = "ACc42426d7aded911a0b366b00b6a7b88c"
    # Your Auth Token from twilio.com/console
    auth_token  = "e210b29c177792f00f12a9ba44ced8e4"

    client = Client(account_sid, auth_token)

    message = client.messages.create(
        to="+9191482 18777", 
        from_="+19286159310",
        body=name + " has entered please take necessary precaution")
    print(message.sid)