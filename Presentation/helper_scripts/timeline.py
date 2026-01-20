import matplotlib.pyplot as plt

# --- Multi-Scale Data ---
data = [
    ("BTC Price (Daily)", 
     [("Jan 1", "74k"), ("Jan 2", "75k"), ("Jan 3", "77k"), ("Jan 4", "78k"), ("Jan 5", "79k"), 
      ("Jan 6", "80k"), ("Jan 7", "79k"), ("Jan 8", "78k"), ("Jan 9", "78k"), ("Jan 10", "78k")]),
    
    ("Innsbruck Temp (Monthly)", 
     [("Jan", "-2°"), ("Feb", "1°"), ("Mar", "7°"), ("Apr", "12°"), ("May", "18°"), 
      ("Jun", "22°"), ("Jul", "24°"), ("Aug", "23°"), ("Sep", "17°"), ("Oct", "11°")]),
    
    ("US President (4-Year)", 
     [("2008", "Obama"), ("2012", "Obama"), ("2016", "Trump"), ("2020", "Biden"), ("2024", "Biden"), 
      ("2025", "Trump"), ("2028", "Trump"), ("2032", "TBD"), ("2036", "TBD"), ("2040", "TBD")]),
    
    ("Halley's Comet (76-Year)", [("1758", "Sighted"),
        ("1835", "Sighted"),
        ("1910", "Not sighted"),
        ("1986", "Sighted"),
        ("2061", "Expected"),
        ("2137", "Expected"),
        ("2213", "Expected"),
        ("2289", "Expected"),
        ("2365", "Expected"),
        ("2441", "Expected")]),
    
    ("Gold Symbol (Constant)", 
     [("1800", "Au"), ("1900", "Au"), ("2000", "Au"), ("2100", "Au"), ("2200", "Au"), 
      ("2300", "Au"), ("2400", "Au"), ("2500", "Au"), ("2600", "Au"), ("2700", "Au")])
]

COLORS = {"point": "#0072C6", "stable": "#D3D3D3", "change": "#E63946"}

def save_step(count):
    fig, ax = plt.subplots(figsize=(14, 8))
    for idx, (label, events) in enumerate(data[:count]):
        y = (len(data) - idx) * 1.5
        ax.text(-0.5, y, label.upper(), ha="right", va="center", fontweight="bold", fontsize=17)

        for i, (date, val) in enumerate(events):
            if i < len(events) - 1:
                changed = val != events[i+1][1]
                ax.plot([i, i+1], [y, y], color=COLORS["change"] if changed else COLORS["stable"], 
                        lw=5, solid_capstyle='round', zorder=1)

            ax.scatter(i, y, s=150, color=COLORS["point"], edgecolors="white", zorder=3)
            ax.text(i, y + 0.35, date, ha="center", fontsize=12, color="gray", rotation=30)
            ax.text(i, y - 0.5, val, ha="center", fontsize=13, fontweight="bold")

    ax.set_xlim(-2.5, 10)
    ax.set_ylim(0, (len(data) + 1) * 1.5)
    ax.axis('off')
    plt.tight_layout(pad=-4)
    plt.savefig(f"volatility_reveal_{count}.pdf")
    plt.close()

for i in range(1, 6):
    save_step(i)