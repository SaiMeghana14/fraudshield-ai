from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime

def create_pdf():
    filename = "fraud_report.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, height - 50, "FraudShield AI Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 100, f"Generated on: {datetime.datetime.now()}")

    c.drawString(100, height - 150, "This report summarizes fraud detection results, anomalies in trades,")
    c.drawString(100, height - 170, "and phishing scam classification. Intended for investor protection.")

    c.save()
    return filename
