x_loan =319900 - 50000
monthly_loan = x_loan/30/12
mortgage_rate = 0.065  

def curIntRate(year):
    return (x_loan - monthly_loan * year) * mortgage_rate
        
total_int = 0
for i in range(0,31):
    total_int = total_int + curIntRate(i)

print((total_int + 319900 - 50000)/12/30)