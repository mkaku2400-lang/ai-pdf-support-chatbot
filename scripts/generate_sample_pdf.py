"""Generate sample_company_faq.pdf, a FAQ for a fictional company, for testing the app.

Usage:  python scripts/generate_sample_pdf.py
"""

from pathlib import Path

from fpdf import FPDF

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "sample_company_faq.pdf"

COMPANY = "BrightNest Home"
INTRO = (
    "BrightNest Home is an online store for smart home devices, small kitchen appliances and "
    "home accessories. This document answers the questions our customers ask most often. "
    "BrightNest Home is a fictional company created for demonstration purposes."
)

FAQ = {
    "Shipping and Delivery": [
        ("How long does shipping take?",
         "Standard shipping takes 3 to 5 business days within the United States. Express shipping "
         "takes 1 to 2 business days. Orders placed before 2:00 PM Eastern Time are dispatched the same day."),
        ("How much does shipping cost?",
         "Standard shipping is free on all orders over $50. For orders under $50, standard shipping "
         "costs $4.99. Express shipping costs a flat $14.99 on any order."),
        ("Do you ship internationally?",
         "Yes. We ship to Canada, the United Kingdom, Australia and most countries in the European Union. "
         "International delivery takes 7 to 14 business days and costs $19.99. Customs duties and import "
         "taxes are paid by the customer."),
        ("How can I track my order?",
         "You will receive a tracking link by email as soon as your order is dispatched. You can also "
         "track your order at any time from the My Orders page of your account."),
    ],
    "Returns and Refunds": [
        ("What is your return policy?",
         "You can return any item within 30 days of delivery for a full refund. The item must be unused "
         "and in its original packaging. Return shipping is free for customers in the United States."),
        ("How do I start a return?",
         "Go to the My Orders page, select the order and click Request Return. We will email you a "
         "prepaid return label within 24 hours."),
        ("When will I receive my refund?",
         "Refunds are issued to your original payment method within 5 to 7 business days after we "
         "receive the returned item. You will get an email confirmation when the refund is processed."),
        ("Which items cannot be returned?",
         "Gift cards, clearance items marked Final Sale, and opened personal care products cannot be returned."),
    ],
    "Warranty and Repairs": [
        ("What warranty do your products have?",
         "All BrightNest smart home devices come with a 2-year limited warranty covering manufacturing "
         "defects. Kitchen appliances carry a 1-year limited warranty. Accidental damage and normal wear "
         "are not covered."),
        ("How do I make a warranty claim?",
         "Email warranty@brightnesthome.example with your order number and a photo or short video of the "
         "problem. Our team replies within 2 business days and, if the claim is approved, sends a "
         "replacement at no cost."),
    ],
    "Orders and Payments": [
        ("Which payment methods do you accept?",
         "We accept Visa, Mastercard, American Express, PayPal, Apple Pay and Google Pay. We also offer "
         "interest-free installments through Klarna on orders over $100."),
        ("Can I change or cancel my order?",
         "You can change or cancel your order within 1 hour of placing it from the My Orders page. After "
         "that the order is prepared for dispatch and can no longer be changed, but you can return it "
         "once it arrives."),
        ("Do you offer discounts for bulk or business orders?",
         "Yes. Businesses ordering 10 or more units of the same product receive a 15 percent discount. "
         "Contact sales@brightnesthome.example for a custom quote."),
        ("Do you have a loyalty program?",
         "Yes. BrightNest Rewards members earn 1 point for every dollar spent. Every 100 points can be "
         "redeemed for a $5 discount. Joining is free."),
    ],
    "Customer Support": [
        ("How can I contact customer support?",
         "You can reach our support team by email at support@brightnesthome.example, by phone at "
         "1-800-555-0142, or by live chat on our website."),
        ("What are your support hours?",
         "Our support team is available Monday to Friday from 9:00 AM to 6:00 PM Eastern Time, and "
         "Saturday from 10:00 AM to 2:00 PM. We are closed on Sundays and public holidays."),
        ("Do you offer installation help for smart home devices?",
         "Yes. Every smart home device includes a free 20-minute video setup call with a BrightNest "
         "specialist. You can book the call from the link in your order confirmation email."),
    ],
}


def build_pdf() -> FPDF:
    pdf = FPDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=20, top=18, right=20)
    pdf.add_page()

    def line(text: str, style: str, size: int, height: float, space_after: float = 0) -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")
        if space_after:
            pdf.ln(space_after)

    pdf.set_text_color(31, 78, 121)
    line(COMPANY, "B", 24, 11)
    pdf.set_text_color(90, 90, 90)
    line("Customer Help Center - Frequently Asked Questions", "", 13, 8, space_after=4)
    pdf.set_text_color(30, 30, 30)
    line(INTRO, "", 11, 6, space_after=4)

    for section, entries in FAQ.items():
        pdf.set_text_color(31, 78, 121)
        pdf.ln(2)
        line(section, "B", 15, 9, space_after=1)
        pdf.set_text_color(30, 30, 30)
        for question, answer in entries:
            line(f"Q: {question}", "B", 11, 6)
            line(f"A: {answer}", "", 11, 6, space_after=3)
    return pdf


if __name__ == "__main__":
    build_pdf().output(str(OUTPUT_PATH))
    print(f"Created {OUTPUT_PATH}")
