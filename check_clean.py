with open('app/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

patterns = ['23CSE301', 'B.Tech', 'Review 1', 'Review 2', 'Rubric', 'Capstone', '+2 Bonus', 'rubric', 'Marks Each', '25 marks', 'bonus mark']
found = []
for p in patterns:
    if p.lower() in content.lower():
        found.append(p)

if found:
    print('Still found:', found)
    for p in found:
        for i, line in enumerate(content.split('\n'), 1):
            if p.lower() in line.lower():
                print(f'  Line {i}: {line.strip()[:100]}')
else:
    print('All clean! No academic references remain.')
