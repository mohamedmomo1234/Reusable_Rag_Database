from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def chat_to_text(messages):
    lines = []

    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}:")
        lines.append(message["content"])
        lines.append("")

    return "\n".join(lines)


def chat_to_pdf(messages):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    x = 40
    y = height - 50

    pdf.setFont("Helvetica", 10)

    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"

        lines = [
            f"{role}:",
            message["content"],
            "",
        ]

        for line in lines:
            words = line.split()
            current = ""

            for word in words:
                test = f"{current} {word}".strip()

                if pdf.stringWidth(test, "Helvetica", 10) > width - 80:
                    pdf.drawString(x, y, current)
                    y -= 14
                    current = word
                else:
                    current = test

            if current:
                pdf.drawString(x, y, current)
                y -= 14

            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 50

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
