import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from model import train_anomaly_model, is_anomaly

# ─── Configuration ──────────────────────────────────────
DAILY_BUDGET   = 250.0
WEEKLY_BUDGET  = 1750.0
MONTHLY_BUDGET = 8000.0

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///expenses.db'
db = SQLAlchemy(app)

# ─── Database Model ─────────────────────────────────────
class Expense(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    amount    = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime,
                          default=datetime.datetime.utcnow)

# create all tables up‐front, as soon as app context is available
with app.app_context():
    db.create_all()

# ─── Routes ─────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    message = None

    if request.method == 'POST':
        # 1. Save new expense
        amt = float(request.form['amount'])
        desc = request.form['description']
        new_exp = Expense(amount=amt, description=desc)
        db.session.add(new_exp)
        db.session.commit()

        # 2. Compute totals
        today = datetime.date.today()
        daily_start = datetime.datetime.combine(today, datetime.time.min)
        week_start  = today - datetime.timedelta(days=today.weekday())
        week_start = datetime.datetime.combine(week_start, datetime.time.min)
        month_start = today.replace(day=1)
        month_start = datetime.datetime.combine(month_start, datetime.time.min)

        daily_total   = sum(e.amount for e in Expense.query.filter(Expense.timestamp >= daily_start).all())
        weekly_total  = sum(e.amount for e in Expense.query.filter(Expense.timestamp >= week_start).all())
        monthly_total = sum(e.amount for e in Expense.query.filter(Expense.timestamp >= month_start).all())

        # 3. Budget check
        if daily_total > DAILY_BUDGET:
            message = f"🚨 You’ve exceeded your **daily** budget of {DAILY_BUDGET}!"
        elif weekly_total > WEEKLY_BUDGET:
            message = f"🚨 You’ve exceeded your **weekly** budget of {WEEKLY_BUDGET}!"
        elif monthly_total > MONTHLY_BUDGET:
            message = f"🚨 You’ve exceeded your **monthly** budget of {MONTHLY_BUDGET}!"
        else:
            message = "✅ You’re within all budgets so far."

        # 4. Very basic AI: anomaly detection
        #    flag if this amount is unusually large vs. your past entries
        records = Expense.query.all()
        df = pd.DataFrame([{'amount': e.amount} for e in records])
        if len(df) > 5:
            mdl = train_anomaly_model(df)
            if is_anomaly(mdl, amt):
                message += "  🤖 AI: This entry looks unusual compared to your past spending."

    # Recompute for display
    today = datetime.date.today()
    daily_start = datetime.datetime.combine(today, datetime.time.min)
    week_start  = today - datetime.timedelta(days=today.weekday())
    week_start = datetime.datetime.combine(week_start, datetime.time.min)
    month_start = today.replace(day=1)
    month_start = datetime.datetime.combine(month_start, datetime.time.min)

    daily_total   = sum(e.amount for e in Expense.query.filter(Expense.timestamp >= daily_start).all())
    weekly_total  = sum(e.amount for e in Expense.query.filter(Expense.timestamp >= week_start).all())
    monthly_total = sum(e.amount for e in Expense.query.filter(Expense.timestamp >= month_start).all())

    all_expenses = Expense.query.order_by(Expense.timestamp.desc()).all()
    budgets = {'daily': DAILY_BUDGET, 'weekly': WEEKLY_BUDGET, 'monthly': MONTHLY_BUDGET}

    return render_template(
        'dashboard.html',
        expenses=all_expenses,
        daily_total=daily_total,
        weekly_total=weekly_total,
        monthly_total=monthly_total,
        budgets=budgets,
        message=message
    )
    
@app.route('/delete/<int:expense_id>', methods=['POST'])
def delete_expense(expense_id):
    exp = Expense.query.get_or_404(expense_id)
    db.session.delete(exp)
    db.session.commit()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
