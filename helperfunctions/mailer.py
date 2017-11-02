# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:23:23 2017

@author: hhofmann
"""

import smtplib
import time

def mailto(msg):
    absender = 'heinzkonto@hotmail.com'
    adressat = 'heinz.hofmann@hotmail.com'
    betreff = 'Mail von DGX'
    inhalt = '\n' + msg
    zeit = time.ctime(time.time())
    text = 'From: ' + absender + '\n'
    text += 'To:' + adressat + '\n'
    text += 'Date:' + zeit + '\n'
    text += 'Subject:' + betreff + '\n'
    
    text += inhalt
    server = smtplib.SMTP('smtp.live.com')
    server.starttls()
    server.login('heinzkonto@hotmail.com','tuttifrutti3000:)')
    server.sendmail(absender,adressat,text)
    server.quit()
    
    print("mail sent")
    
if __name__ == "__main__":
#==============================================================================
#     texttosend = "Hallo Heinz Hofmann\n\n"
#     texttosend += "Mein Name ist DGX und ich möchte dir folgendes mitteilen: \n"
#     texttosend += "Ein Mail direkt aus einem Python-Programm zu versenden ist extrem einfach \n\n"
#     texttosend += "Dafür einfach die library mailer importieren und mailer.mailto(text) aufrufen\n\n"
#     texttosend += "liebe Grüsse\n"
#     texttosend += "DGX 1"
#     mailto(texttosend)
#==============================================================================
    x=10
    batchSize = 9
    learning_rate = 5
    num_threads = 3
    min_after_dequeue = 20
    origin_path = "origin path"
    i = 2000
    
    print("finished")
    mailtext = " finished \n"
    mailtext += "\n cost                = " + str(x)
    mailtext += "\n batchSize           = " + str(batchSize)
    mailtext += "\n learningrate        = " + str(learning_rate)
    mailtext += "\n num_threads         = " + str(num_threads)
    mailtext += "\n min_after_dequeue   = " + str(min_after_dequeue)
    mailtext += "\n trained on          = " + origin_path
    mailtext += "\n number of iterations= " + str(i)

    mailto(mailtext)
    
