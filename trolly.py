import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# 가치 판단 요소(3) + 결과(1)
x_number = np.arange(0, 11, 1)
x_intervention = np.arange(0, 11, 1)
x_age = np.arange(0, 11, 1)

x_result = np.arange(0, 2, 0.1)

# 퍼지 소속 함수
number_lo = fuzz.trimf(x_number, [0, 0, 5])
number_md = fuzz.trimf(x_number, [0, 5, 10])
number_hi = fuzz.trimf(x_number, [5, 10, 10])
intervention_lo = fuzz.trimf(x_intervention, [0, 0, 5])
intervention_md = fuzz.trimf(x_intervention, [0, 5, 10])
intervention_hi = fuzz.trimf(x_intervention, [5, 10, 10])
age_lo = fuzz.trimf(x_age, [0, 0, 5])
age_md = fuzz.trimf(x_age, [0, 5, 10])
age_hi = fuzz.trimf(x_age, [5, 10, 10])

result_1 = fuzz.trimf(x_result, [0, 0, 1])
result_2 = fuzz.trimf(x_result, [0, 1, 1])


# Visualization
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_number, number_lo, 'b', linewidth=1.5, label='low')
ax0.plot(x_number, number_md, 'g', linewidth=1.5, label='medium')
ax0.plot(x_number, number_hi, 'r', linewidth=1.5, label='high')
ax0.set_title('number_factor')
ax0.legend()

ax1.plot(x_intervention, intervention_lo, 'b', linewidth=1.5, label='low')
ax1.plot(x_intervention, intervention_md, 'g', linewidth=1.5, label='medium')
ax1.plot(x_intervention, intervention_hi, 'r', linewidth=1.5, label='high')
ax1.set_title('intervention_factor')
ax1.legend()

ax2.plot(x_age, age_lo, 'b', linewidth=1.5, label='low')
ax2.plot(x_age, age_md, 'g', linewidth=1.5, label='medium')
ax2.plot(x_age, age_hi, 'r', linewidth=1.5, label='high')
ax2.set_title('age_factor')
ax2.legend()

fig, ax3 = plt.subplots(figsize=(8, 9))
ax3.plot(x_result, result_1, 'b', linewidth=1.5, label='1')
ax3.plot(x_result, result_2, 'g', linewidth=1.5, label='2')

# Turn off top/right axes
for ax in (ax0, ax1, ax2, ax3):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
#plt.show()

try:
    Mydata_number, Mydata_intervention, Mydata_age = map(float, input("희생자 숫자의 중요도, 개입에 대한 회피 선호도, 연령 선호도를 입력하세요(0-10): ").split())
except ValueError as e:
    print("숫자 형식이 아닙니다. 에러 메시지:", e)
    # 예외 처리 후 프로그램 종료 또는 다른 조치를 취할 수 있음
    exit()

#그룹 1 정보 입력받기
try:
    group1num = int(input("그룹 1의 사람 수를 입력하세요: "))
    user_input_str = input("사람들의 나이를 입력하세요 (예: 1 2 3): ")
    group1age = [int(item.strip()) for item in user_input_str.split(' ')]
    
    # 입력받은 숫자들의 평균 계산
    average_age_1 = sum(group1age) / len(group1age)

except ValueError as e:
    print("올바른 리스트 형식이 아닙니다. 에러 메시지:", e)

#그룹 2 정보 입력받기
try:
    group2num = int(input("그룹 2의 사람 수를 입력하세요: "))
    user_input_str = input("사람들의 나이를 입력하세요 (예: 1 2 3): ")
    group2age = [int(item.strip()) for item in user_input_str.split(' ')]
    
    # 입력받은 숫자들의 평균 계산
    average_age_2 = sum(group2age) / len(group2age)

except ValueError as e:
    print("올바른 리스트 형식이 아닙니다. 에러 메시지:", e)

if(len(group1age) <= len(group2age)):
    more_people = result_2
else : 
    more_people = result_1

if(average_age_1 <= average_age_2):
    younger_people = result_1
else : 
    younger_people = result_2


# We need the activation of our fuzzy membership functions at these values.
# The exact values 6.5 and 9.8 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
number_level_lo = fuzz.interp_membership(x_number, number_lo, Mydata_number)
number_level_md = fuzz.interp_membership(x_number, number_md, Mydata_number)
number_level_hi = fuzz.interp_membership(x_number, number_hi, Mydata_number)

intervention_level_lo = fuzz.interp_membership(x_intervention, intervention_lo, Mydata_intervention)
intervention_level_md = fuzz.interp_membership(x_intervention, intervention_md, Mydata_intervention)
intervention_level_hi = fuzz.interp_membership(x_intervention, intervention_hi, Mydata_intervention)

age_level_lo = fuzz.interp_membership(x_age, age_lo, Mydata_age)
age_level_md = fuzz.interp_membership(x_age, age_md, Mydata_age)
age_level_hi = fuzz.interp_membership(x_age, age_hi, Mydata_age)


result_activation_1 = np.fmin(number_level_hi, more_people)  # removed entirely to 0

# For rule 2 we connect acceptable service to medium tipping
result_activation_2 = np.fmin(intervention_level_hi, result_1)

# For rule 3 we connect high service OR high food with high tipping
result_activation_3 = np.fmin(age_level_hi, younger_people)

# Aggregate all three output membership functions together
aggregated = np.fmax(result_activation_1,
                     np.fmax(result_activation_2, result_activation_3))

x_result_array = np.arange(0, 2, 0.1)

# Calculate defuzzified result
result = fuzz.defuzz(x_result_array, aggregated, 'centroid')
rounded_result = round(result*100, 1)

if(rounded_result<50):
    choose = "1번 선택지"
    rounded_result = 100-rounded_result
else : 
    choose = "2번 선택지"

# 뒷부분에 문자열을 붙여 출력
result_string = f"{rounded_result}의 확률로 {choose}를 선택할 것입니다."
print(result_string)


#visualization
result_activation = fuzz.interp_membership(x_result, aggregated, result)  # for plot

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_result, result_1, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_result, result_2, 'g', linewidth=0.5, linestyle='--')

tip0 = np.zeros_like(x_result)

ax0.fill_between(x_result, tip0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([result, result], [0, result_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()