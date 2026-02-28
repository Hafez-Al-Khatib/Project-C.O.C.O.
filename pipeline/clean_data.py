"""
Project C.O.C.O. — Data Cleaning Pipeline
==========================================
Cleans all raw CSV report exports from the Conut bakery POS system.
Strips pagination headers, normalises numbers, and outputs clean parquet files.
"""

import os
import re
import pandas as pd
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Conut bakery Scaled Data")
OUT_DIR = os.path.join(BASE_DIR, "cleaned")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_number(val):
    """Parse comma-formatted numbers like '1,251,486.48' into float."""
    if pd.isna(val) or val == "":
        return 0.0
    s = str(val).strip().replace('"', "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def is_page_header(line_str):
    """Detect repeated pagination header lines in POS CSV exports."""
    if re.search(r"Page\s+\d+\s+of", line_str, re.IGNORECASE):
        return True
    return False


# ---------------------------------------------------------------------------
# 1. REP_S_00502 — Sales by customer (co-purchase baskets)
# ---------------------------------------------------------------------------
def clean_transactions():
    """Parse REP_S_00502.csv into customer purchase baskets per branch."""
    filepath = os.path.join(DATA_DIR, "REP_S_00502.csv")
    rows = []
    current_branch = None
    current_customer = None
    receipt_counter = 0

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or is_page_header(line_stripped):
                continue
            # Skip column header lines
            if line_stripped.startswith("Full Name,"):
                continue
            # Skip report title lines
            if line_stripped.startswith("Conut - Tyre,,,,") and "Sales by customer" not in line_stripped:
                if "Sales by customer" not in line_stripped:
                    pass
            if "Sales by customer" in line_stripped:
                continue
            if line_stripped.startswith("30-Jan-26"):
                continue

            parts = line.split(",")

            # Detect branch header
            branch_match = re.match(r"^Branch\s*:\s*(.+?)(?:,|$)", line_stripped)
            if branch_match:
                current_branch = branch_match.group(1).strip()
                continue

            # Detect total branch line
            if line_stripped.startswith("Total Branch:"):
                continue

            # Detect customer name (non-empty first column)
            first_col = parts[0].strip() if parts else ""

            # Skip total lines
            if first_col.startswith("Total :") or first_col.startswith("Total:"):
                continue

            # Skip footer
            if first_col.startswith("REP_S_"):
                continue

            # Customer name line: first col is non-empty and not a number
            if first_col and not first_col.replace(".", "").replace("-", "").lstrip("0").isdigit():
                # Could be "Person_XXXX" or "0 Person_XXXX" etc.
                name_match = re.match(r"^(?:\d+\s+)?(Person_\d+)", first_col)
                if name_match:
                    current_customer = name_match.group(1)
                    receipt_counter += 1  # New customer block = new receipt/visit
                    # Check if this line also has item data
                    if len(parts) >= 4 and parts[1].strip():
                        qty = parse_number(parts[1])
                        desc = parts[2].strip().strip('"').strip()
                        price = parse_number(parts[3]) if len(parts) > 3 else 0.0
                        if qty > 0 and desc:
                            rows.append({
                                "receipt_id": receipt_counter,
                                "branch": current_branch,
                                "customer": current_customer,
                                "qty": qty,
                                "item": desc,
                                "price": price,
                            })
                    continue

            # Item line: first col is empty, qty in second col
            if not first_col and len(parts) >= 4:
                qty = parse_number(parts[1])
                # Description may contain commas inside quotes
                desc = parts[2].strip().strip('"').strip()
                price = parse_number(parts[3]) if len(parts) > 3 else 0.0

                if qty > 0 and desc and current_customer:
                    rows.append({
                        "receipt_id": receipt_counter,
                        "branch": current_branch,
                        "customer": current_customer,
                        "qty": qty,
                        "item": desc,
                        "price": price,
                    })

    df = pd.DataFrame(rows)

    # Normalize item names
    df["item"] = df["item"].str.strip().str.strip(".").str.strip(",").str.strip()
    df["item"] = df["item"].str.replace(r"\s+", " ", regex=True)
    # Remove modifier/add-on items (price == 0 and look like options)
    # Keep only real purchasable menu items (price > 0) for graph analysis
    df_products = df[df["price"] > 0].copy()

    outpath = os.path.join(OUT_DIR, "transactions.parquet")
    df.to_parquet(outpath, index=False)
    df_products.to_parquet(os.path.join(OUT_DIR, "transactions_products.parquet"), index=False)
    print(f"[✓] Transactions: {len(df)} total rows, {len(df_products)} product rows → {outpath}")
    return df, df_products


# ---------------------------------------------------------------------------
# 2. rep_s_00191_SMRY — Sales by Items by Group (per branch)
# ---------------------------------------------------------------------------
def clean_sales_by_item():
    """Parse rep_s_00191_SMRY.csv into item-level sales per branch."""
    filepath = os.path.join(DATA_DIR, "rep_s_00191_SMRY.csv")
    rows = []
    current_branch = None
    current_division = None
    current_group = None

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or is_page_header(line_stripped):
                continue
            if line_stripped.startswith("30-Jan-26"):
                continue
            if line_stripped.startswith("Description,"):
                continue
            if "Sales by Items" in line_stripped:
                continue
            if line_stripped.startswith("REP_S_") or line_stripped.startswith("rep_s_"):
                continue

            # Detect branch
            branch_match = re.match(r"^Branch:\s*(.+?)(?:,|$)", line_stripped)
            if branch_match:
                current_branch = branch_match.group(1).strip()
                continue

            # Title line (first line of a new branch section)
            if line_stripped.startswith("Conut") and ",," in line_stripped:
                # e.g. "Conut - Tyre,,,,"
                parts = line_stripped.split(",")
                if parts[0].strip() and all(p.strip() == "" for p in parts[1:4]):
                    continue

            parts = line.split(",")
            first_col = parts[0].strip().strip('"')

            # Detect Total lines
            if first_col.startswith("Total by"):
                continue

            # Detect Division
            div_match = re.match(r"^Division:\s*(.+?)(?:,|$)", line_stripped)
            if div_match:
                current_division = div_match.group(1).strip()
                continue

            # Detect Group
            grp_match = re.match(r"^Group:\s*(.+?)(?:,|$)", line_stripped)
            if grp_match:
                current_group = grp_match.group(1).strip()
                continue

            # Item line: Description,,Qty,Total Amount,
            if first_col and len(parts) >= 4:
                desc = first_col
                qty = parse_number(parts[2]) if len(parts) > 2 else 0.0
                amount = parse_number(parts[3]) if len(parts) > 3 else 0.0

                if qty > 0 or amount > 0:
                    rows.append({
                        "branch": current_branch,
                        "division": current_division,
                        "group": current_group,
                        "item": desc,
                        "qty": qty,
                        "total_amount": amount,
                    })

    df = pd.DataFrame(rows)
    outpath = os.path.join(OUT_DIR, "sales_by_item.parquet")
    df.to_parquet(outpath, index=False)
    print(f"[✓] Sales by item: {len(df)} rows, {df['branch'].nunique()} branches → {outpath}")
    return df


# ---------------------------------------------------------------------------
# 3. rep_s_00334_1_SMRY — Monthly Sales by Branch
# ---------------------------------------------------------------------------
def clean_monthly_sales():
    """Parse rep_s_00334_1_SMRY.csv into monthly sales per branch."""
    filepath = os.path.join(DATA_DIR, "rep_s_00334_1_SMRY.csv")
    rows = []
    current_branch = None

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or is_page_header(line_stripped):
                continue
            if line_stripped.startswith("30-Jan-26") or line_stripped.startswith("Month,"):
                continue
            if "Monthly Sales" in line_stripped:
                continue
            if line_stripped.startswith("REP_S_") or line_stripped.startswith("rep_s_"):
                continue
            if line_stripped.startswith("Conut - Tyre,,,,") and "Monthly" not in line_stripped:
                continue

            parts = line.split(",")
            first_col = parts[0].strip().strip('"')

            # Branch header
            branch_match = re.match(r"^Branch Name:\s*(.+?)(?:,|$)", line_stripped)
            if branch_match:
                current_branch = branch_match.group(1).strip()
                continue

            # Total lines
            if "Total" in first_col:
                continue
            if "Grand Total" in line_stripped:
                continue

            # Month data line: Month,,Year,Total,
            month_names = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            if first_col in month_names and current_branch:
                year = parse_number(parts[2]) if len(parts) > 2 else 2025
                total = parse_number(parts[3]) if len(parts) > 3 else 0.0
                month_num = month_names.index(first_col) + 1
                rows.append({
                    "branch": current_branch,
                    "month": month_num,
                    "month_name": first_col,
                    "year": int(year),
                    "total_sales": total,
                })

    df = pd.DataFrame(rows)
    outpath = os.path.join(OUT_DIR, "monthly_sales.parquet")
    df.to_parquet(outpath, index=False)
    print(f"[✓] Monthly sales: {len(df)} rows → {outpath}")
    return df


# ---------------------------------------------------------------------------
# 4. REP_S_00461 — Time & Attendance
# ---------------------------------------------------------------------------
def parse_time(time_str):
    """Parse time in HH.MM.SS format to total seconds."""
    parts = time_str.strip().split(".")
    if len(parts) == 3:
        try:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + s
        except ValueError:
            return 0
    return 0


def parse_duration(dur_str):
    """Parse work duration like '08.36.34' to hours as float."""
    parts = dur_str.strip().split(".")
    if len(parts) == 3:
        try:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h + m / 60 + s / 3600
        except ValueError:
            return 0.0
    return 0.0


def clean_labor_hours():
    """Parse REP_S_00461.csv into labor hours per employee per day."""
    filepath = os.path.join(DATA_DIR, "REP_S_00461.csv")
    rows = []
    current_employee = None
    current_emp_id = None
    current_branch = None

    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or is_page_header(line_stripped):
                continue
            if "Time & Attendance" in line_stripped:
                continue
            if line_stripped.startswith("REP_S_") or line_stripped.startswith("rep_s_"):
                continue
            if line_stripped.startswith(",30-Jan-26"):
                continue
            if ",PUNCH IN,,PUNCH OUT,,Work Duration" in line_stripped:
                continue

            parts = line.split(",")
            first_col = parts[0].strip()

            # Employee header: ,EMP ID :X.0,NAME :Person_XXXX,,,
            emp_match = re.search(r"EMP ID\s*:\s*([\d.]+)\s*.*NAME\s*:\s*(Person_\d+)", line_stripped)
            if emp_match:
                current_emp_id = emp_match.group(1)
                current_employee = emp_match.group(2)
                continue

            # Branch line under employee
            branch_names = ["Conut - Tyre", "Conut Jnah", "Main Street Coffee", "Conut"]
            for bn in branch_names:
                if f",{bn},," in line_stripped or line_stripped.strip(",") == bn:
                    current_branch = bn
                    break

            # Total line
            if "Total :" in line_stripped:
                continue

            # Shift data line: date,,punch_in,date,,punch_out,duration
            date_match = re.match(r"^(\d{2}-\w{3}-\d{2})", first_col)
            if date_match and current_employee:
                punch_date = first_col
                # Duration is typically the last non-empty field
                duration_str = parts[-1].strip() if parts else ""
                if not duration_str:
                    # Try second to last
                    duration_str = parts[-2].strip() if len(parts) > 1 else ""

                duration_hours = parse_duration(duration_str)

                # Skip trivial punches (< 0.01 hours = 36 seconds)
                if duration_hours >= 0.01:
                    rows.append({
                        "employee_id": current_emp_id,
                        "employee_name": current_employee,
                        "branch": current_branch,
                        "date": punch_date,
                        "work_hours": round(duration_hours, 2),
                    })

    df = pd.DataFrame(rows)
    # Parse dates properly
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y", errors="coerce")
    outpath = os.path.join(OUT_DIR, "labor_hours.parquet")
    df.to_parquet(outpath, index=False)
    print(f"[✓] Labor hours: {len(df)} shifts, {df['employee_name'].nunique()} employees → {outpath}")
    return df


# ---------------------------------------------------------------------------
# 5. rep_s_00435_SMRY — Average Sales by Menu
# ---------------------------------------------------------------------------
def clean_avg_sales():
    """Parse rep_s_00435_SMRY.csv into item-level average sales."""
    filepath = os.path.join(DATA_DIR, "rep_s_00435_SMRY.csv")
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or is_page_header(line_stripped):
                continue
            if line_stripped.startswith("REP_S_") or line_stripped.startswith("rep_s_"):
                continue
            # Skip header row if present
            if "Item" in line_stripped and "Avg" in line_stripped:
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                item_name = parts[0].strip().strip('"')
                avg_price = parse_number(parts[1]) if len(parts) > 1 else 0.0
                total_qty = parse_number(parts[2]) if len(parts) > 2 else 0.0
                if item_name:
                    rows.append({
                        "item": item_name,
                        "avg_price": avg_price,
                        "total_quantity": total_qty
                    })

    if rows:
        df = pd.DataFrame(rows)
        outpath = os.path.join(OUT_DIR, "avg_sales_menu.parquet")
        df.to_parquet(outpath, index=False)
        print(f"[✓] Avg sales menu: {len(df)} rows, {df['item'].nunique()} items → {outpath}")
        return df
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_all():
    print("=" * 60)
    print("  Project C.O.C.O. — Data Cleaning Pipeline")
    print("=" * 60)

    print("\n[1/5] Cleaning transaction data (REP_S_00502)...")
    clean_transactions()

    print("\n[2/5] Cleaning sales by item data (rep_s_00191_SMRY)...")
    clean_sales_by_item()

    print("\n[3/5] Cleaning monthly sales data (rep_s_00334_1_SMRY)...")
    clean_monthly_sales()

    print("\n[4/5] Cleaning labor hours data (REP_S_00461)...")
    clean_labor_hours()

    print("\n[5/5] Cleaning avg sales menu (rep_s_00435_SMRY)...")
    clean_avg_sales()

    print("\n" + "=" * 60)
    print("  Data cleaning complete! Outputs in ./cleaned/")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
