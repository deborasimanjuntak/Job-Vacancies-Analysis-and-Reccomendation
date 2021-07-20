#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets
from ipywidgets import interact
import warnings
warnings.filterwarnings('ignore')


# In[2]:


plt.style.use('seaborn-darkgrid')


# In[3]:


#Importing the dataset
data=pd.read_csv('21. naukri.csv')
data.head(3)


# In[4]:


data.info()


# In[5]:


# There are 14 columns in dataset. Some of them dont have meaningful information, so we are gonna delete these columns
drop=['jobdescription','jobid','site_name','uniq_id','jobtitle','postdate']
data=data.drop(columns=drop)
data.head()


# ## Cleaning and Analysing Column Experience ##

# In[6]:


data.experience.isnull().sum()


# In[7]:


data.experience.value_counts()[:20]


# In[8]:


# There are 4 missing values in column experience, I will fill the missing value with the mode of this column
data.experience.fillna("2 - 7 yrs",inplace=True)


# In[9]:


# Here I split the data before extracting the minimum and maximum year experience
year_experience=data.experience.str.split(' ')
data['min_year_exp']=year_experience.apply(lambda x:x[0])
data['max_year_exp']=year_experience.apply(lambda x:x[2] if len(x)>2 else x[0])


# In[10]:


data['min_year_exp'].value_counts()


# In[11]:


# The are some rows with value "Not" that need to be replaced
# But before cleaning the data, I want to see what word 'Not' means, so I will go back to columns experience and see the whole row with value "Not"
dirty_list=[]
for i in range(len(data)):
    if ('Not' in data.experience[i]):
        dirty_list.append(data.experience[i])
dirty_list


# In[12]:


# Because its not mentioned, so I am going to replace 'Not' with '2'
data['min_year_exp']=np.where(data.min_year_exp=='Not','2',data.min_year_exp)
data['min_year_exp'].value_counts()


# In[13]:


# We are going to do the same with column max_year_exp
data['max_year_exp'].value_counts()


# In[14]:


# There are rows with value 'Not' and '-1'. I will replace value 'Not' with '7' and value '-1' with 1
data['max_year_exp']=np.where(data.max_year_exp=='Not','7',data.max_year_exp)
data['max_year_exp']=np.where(data.max_year_exp=='-1','1',data.max_year_exp)
data['max_year_exp'].value_counts()


# In[15]:


# Now lets change string in column min_year_exp and max_year_exp into integer
data['min_year_exp']=data['min_year_exp'].astype(int)
data['max_year_exp']=data['max_year_exp'].astype(int)


# In[16]:


# After cleaning column min_year_exp and max_year_exp, I am going to see the distribution in both columns
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.countplot(data.min_year_exp,palette='rocket')
plt.xticks(range(0,20))
plt.title("Distribution of Minimal Year Experience",weight='bold',size=18)
plt.subplot(1,2,2)
sns.countplot(data.max_year_exp,palette='rocket')
plt.xticks(range(0,26))
plt.title("Distribution of Maximal Year Experience",weight='bold',size=18)
plt.show()


# From the distribution above, we see that the most common minimal year experience required for job vacancies is 2 years experience, where the most common maximal year experience required is 5 years experience

# ## Cleaning and Analysing Column Education ##

# In[17]:


data.education.unique()[:5]


# In[18]:


#Check the number of null values in column education
data.education.isnull().sum()


# In[19]:


#I will replace null values with "'UG: Any Graduate - Any Specialization'"
data.education.fillna('UG: Any Graduate - Any Specialization',inplace=True)


# In[20]:


# Here I want to know the most needed Bachelor Degrees by companies
# First, replace "PG" and "Doctorate" with delimiter "|" so I can split the value with delimiter. 
# Then I extract the first value in the list which is Bachelor Degree required for the job
data_edu=data.education.apply(lambda x:x.replace(" PG:","|")).apply(lambda x:x.replace(" Doctorate:","|"))


# In[21]:


data_edu=data_edu.apply(lambda x:x.split("|"))


# In[22]:


data_edu=data_edu.apply(lambda x:x[0])


# In[23]:


data_edu.value_counts()[:10]


# In[24]:


# From the data extracted above, we see that some data has the same meaning but has different name
#so I am going to replace it with the same name
data_edu=data_edu.apply(lambda x:x.replace("UG: Any Graduate - Any Specialization","UG: Any Graduate"))
data_edu=data_edu.apply(lambda x:x.replace("UG: Any Graduate - Any Specialization, Graduation Not Required","UG: Any Graduate"))
data_edu=data_edu.apply(lambda x:x.replace("UG: Any Graduate, Graduation Not Required","UG: Any Graduate"))
data_edu=data_edu.apply(lambda x:x.replace("B.Tech/B.E. - Any Specialization","UG: B.Tech/B.E."))
data_edu=data_edu.apply(lambda x:x.replace("UG: UG: B.Tech/B.E.","UG: B.Tech/B.E."))
data_edu=data_edu.apply(lambda x:x.replace("UG: Graduation Not Required","UG: Any Graduate"))
data_edu=data_edu.apply(lambda x:x.replace("UG: Any Graduate, UG: B.Tech/B.E.","UG: B.Tech/B.E."))
data_edu=data_edu.apply(lambda x:x.replace("UG: B.Tech/B.E., Computers","UG: B.Tech/B.E. - Computers"))
data_edu=data_edu.apply(lambda x:x.replace("UG: B.Tech/B.E., Computers","UG: B.Tech/B.E. - Computers"))
data_edu=data_edu.apply(lambda x:x.replace("UG: B.Com - Commerce","UG: B.Com"))
data_edu=data_edu.apply(lambda x:x.replace("UG: ",""))


# In[25]:


data["degree"]=data_edu
top_10_degree=pd.DataFrame(data.degree.value_counts()[:10])
top_10_degree


# In[26]:


plt.figure(figsize=(12,8))
sns.barplot(y=top_10_degree.index,x=top_10_degree.degree,palette='flare')
plt.title("10 Most In-Demand Degrees in Job Vacancies",weight='bold',size=18)
plt.xticks(size=12,weight='bold')
plt.yticks(size=12,weight='bold')
plt.show()


# From the chart above, we see that most jobs available require degree in any graduate/spesialization. These jobs may not need specific spesialization so they are open to any degree. While for jobs with specific spesialization, most needed bachelor degrees are Bachelor of Technology, Bachelor of Engineering, and Bachelor of Commerce.

# ## Cleaning and Analysing Column Industry ##

# In[27]:


# Check the number of null values in column industy
data.industry.isnull().sum()


# In[28]:


data[data.industry.isnull()]


# In[29]:


# We will fill the nan values in column industry with the mode
data.industry.fillna(data.industry.mode()[0],inplace=True)
data.industry.isnull().sum()


# In[30]:


# Here we are going to display the 15 most popular industries in Job Vacancies
display(pd.DataFrame(data.industry.value_counts()[:15]).style.background_gradient(cmap="Greys"))


# With more than 9000 jobs available, we can see how popular indusrty in Technology is. The number is significantly higher than any other industries, which atmost only reach 1322. The second most popular industry is Education / Teaching / Training and followed by BPO / Call Centre / ITES.

# ## Cleaning and Analysing Column Job Address ##

# In[31]:


data.joblocation_address.head()


# In[32]:


data.joblocation_address.value_counts()[:5]


# In[33]:


data.joblocation_address.isnull().sum()


# In[34]:


# I am going to fill the nan values with "Not Mentioned". 
# The reason why I dont replace it with mode is because the number of null values is pretty high and it can be diverse
# I dont want this to affet the point of our analysis
data.joblocation_address.fillna('Not Mentioned',inplace=True)


# In[35]:


# There are some addresses that have the same meaning but different name, "Bengalore" and "Bengaluru" 
# So we are going to fix it
data['loc']=data.joblocation_address
data['loc']=data['loc'].apply(lambda x:x.replace("Bengaluru/Bangalore","Bengaluru"))
data['loc']=data['loc'].apply(lambda x:x.replace("Bangalore","Bengaluru"))
data['loc']=data['loc'].apply(lambda x:x.replace("Bengaluru/Bangalore , Bengaluru / Bangalore","Bengaluru"))
data['loc']=data['loc'].apply(lambda x:x.replace("Mumbai , Mumbai","Mumbai"))
data['loc']=data['loc'].apply(lambda x:x.replace("Delhi , Delhi","Delhi"))
data['loc']=data['loc'].apply(lambda x:x.replace("Delhi/NCR(National Capital Region)","Delhi"))
data['loc']=data['loc'].apply(lambda x:x.replace("Delhi/NCR(National Capital Region)","Delhi"))
data['loc']=data['loc'].apply(lambda x:x.replace("Noida , Noida/Greater Noida","Noida"))
data['loc']=data['loc'].apply(lambda x:x.replace("Gurgaon , Gurgaon","Gurgaon"))
data['loc']=data['loc'].apply(lambda x:x.replace("Bengaluru , Bengaluru / Bangalore","Bengaluru"))
data['loc']=data['loc'].apply(lambda x:x.replace("Hyderabad / Secunderabad , Hyderabad/Secunderabad","Hyderabad / Secunderabad"))
data['loc']=data['loc'].apply(lambda x:x.replace("Hyderabad / Secunderabad","Hyderabad"))
data['loc']=data['loc'].apply(lambda x:x.replace("Delhi NCR","Delhi"))
data['loc']=data['loc'].apply(lambda x:x.replace("Bengaluru / Bangalore","Bengaluru"))
data['loc']=data['loc'].apply(lambda x:x.replace("Delhi/NCR","Delhi"))
data['loc']=data['loc'].apply(lambda x:x.replace("Greater Noida","Noida"))
data['loc']=data['loc'].apply(lambda x:x.replace("Greater Noida","Noida"))
data['loc']=data['loc'].apply(lambda x:x.replace('Noida/Noida',"Noida"))
data['loc']=data['loc'].apply(lambda x:x.replace("mumbai","Mumbai"))
data['loc']=data['loc'].apply(lambda x:x.replace("Hyderabad/Secunderabad","Hyderabad"))
data['loc']=data['loc'].apply(lambda x:x.replace("Bengaluru / Bengaluru","Bengaluru"))


# In[36]:


# We see that there are some jobs that available in more than 1 location
# So before doing visualization, we we want to split the value in this column 
address=[]
data['loc']=data['loc'].apply(lambda x:x.replace(" , ",", "))
data['loc']=data['loc'].apply(lambda x:x.split(", "))
data['loc']


# In[37]:


# Next we are going to extract each location in each row with function explode
x=data['loc'].explode()
top_15_location=pd.DataFrame(x.value_counts()[:15])
top_15_location.columns=["Count"]

plt.figure(figsize=(12,8))
sns.barplot(y=top_15_location.index,x=top_15_location.Count,palette="icefire")
plt.title("15 Cities with Highest Job Opportunities in India",weight='bold', size=18)
plt.xticks(size=12,weight='bold')
plt.yticks(size=12,weight='bold')
plt.show()


# From the result above, we see that Bengaluru is the city with highest job opportunites in India, followed by Mumbai, Hyderabad, and Delhi. The result is not surprising considering those cities are the most populated cities in India.

# ## Cleaning and Analysing Column Number of Positions ##

# In[38]:


data.numberofpositions.value_counts()


# In[39]:


# Check the number of null values
data.numberofpositions.isnull().sum()


# In[40]:


# It appears that there are so many null values in this columns. This may be because only 1 job available for the position. 
# So we are going to fill null values with 1

data.numberofpositions.fillna(1,inplace=True)
data.numberofpositions=data.numberofpositions.astype(int)
data.numberofpositions


# In[41]:


number_of_position=pd.DataFrame(data.numberofpositions)
number_of_position.numberofpositions=np.where((number_of_position.numberofpositions>=10) & (number_of_position.numberofpositions<50),49,number_of_position.numberofpositions)# Here I am temporary replace value in range [10,49] with 49, later I will replace it with "[10,49]"
number_of_position.numberofpositions=np.where((number_of_position.numberofpositions>=50) & (number_of_position.numberofpositions<100),99,number_of_position.numberofpositions)# Here I am temporary replace value in range [50,99] with 99, later I will replace it with "[50,99]"
number_of_position.numberofpositions=np.where((number_of_position.numberofpositions>=100) & (number_of_position.numberofpositions<1000),101,number_of_position.numberofpositions) # Here I am temporary replace value in range [100,999] with 101, later I will replace it with "[100-999]"
number_of_position.numberofpositions=np.where((number_of_position.numberofpositions>1000) | (number_of_position.numberofpositions==1000),1000,number_of_position.numberofpositions)# Here I am temporary replace value >=1000 with 1000, later I will replace it with "[1000,]"
number_of_position.numberofpositions=number_of_position.numberofpositions.astype(int)
number_of_position.numberofpositions=number_of_position.numberofpositions.apply(lambda x:str(x).replace("49","[10,49]"))
number_of_position.numberofpositions=number_of_position.numberofpositions.apply(lambda x:x.replace("99","[50,99]"))
number_of_position.numberofpositions=number_of_position.numberofpositions.apply(lambda x:x.replace("101","[100,999]"))
number_of_position.numberofpositions=number_of_position.numberofpositions.apply(lambda x:x.replace("1000","[1000,]"))


# In[42]:


plt.figure(figsize=(12,8))
number_of_position.numberofpositions.value_counts().plot(kind='bar',color='lightcoral')
plt.xticks(rotation=360,size=12,weight='bold')
plt.yticks(size=12,weight='bold')
plt.title("Distribution of Number of Positions in Job Vacancies",size=18,weight='bold')
plt.show()


# From the graph above, we see that the majority of job vacancies only open for 1 position. But there is uncertainty about this number because we fiiled over than 17500 missing values with 1.

# ## Cleaning and Analysing Column Skills ##

# In[43]:


data.skills.value_counts()[:10]


# In[44]:


data.skills.isnull().sum()


# In[45]:


# Let see the data that have missing value in columns skills
data[data.skills.isnull()][:10]


# In[46]:


# Lets take a look on column industry in the data above. We see that the industry is pretty diverse.
# So, instead of filling the missing value with the mode of the entire data, I am going to impute skills with the mode of skills of related industry
mode_skills=[]
for industry in data.industry.unique():
    mode=data[data['industry']==industry][['skills','company']].groupby(['skills']).count().sort_values(by='company',ascending=False).index[0]
    mode_skills.append(mode)


# In[47]:


zip_iterator=zip(data.industry.unique().tolist(),mode_skills)
dict_ind_skills=dict(zip_iterator)
dict_ind_skills


# In[48]:


# Now we get the dictionary that consist of industry and the most needed skills for the industry
for i in range(len(data)):
    if pd.isnull(data.skills[i]):
        data.skills[i]=dict_ind_skills[data.industry[i]]


# In[49]:


data.skills.isnull().sum()


# In[50]:


# Now we successfully impute the data, we are going to fetch the 20 most needed skills 
top_20_skills=pd.DataFrame(data['skills'].value_counts()[:20])
top_20_skills.columns=['Count']
plt.figure(figsize=(12,8))
sns.barplot(y=top_20_skills.index,x=top_20_skills.Count,palette='magma')
plt.title("20 Most Needed Skills in Today's Industries",weight='bold',size=18)
plt.show()


# From the chart above, we see that IT Software - Application Programming, Sales and ITES are the most needed skills in today's industry

# ## Cleaning and Analysing Column Payrate ##

# In[51]:


data.payrate[:10]


# In[52]:


data.payrate.isnull().sum()


# In[53]:


# For now, we wil fill nan value with " Not Disclosed by Recruiter"
data.payrate.fillna("Not Disclosed by Recruiter",inplace=True)
data.payrate.value_counts()


# In[54]:


# We see that majority of job vacancies dont display the payrate for job position, let see the percentage
print("The percentage of job vacancies with not pay rate information is:",data.payrate.value_counts()[0]/len(data.payrate))


# In[55]:


# The percentage is really high, so we wont do further analysis in this column


# ## Cleaning and Analysing Column Company ##

# In[56]:


data.company.value_counts()[:10]


# In[57]:


data.company.isnull().sum()


# In[58]:


# We are going to impute missing value with "Confidential"
data.company.fillna("Confidential",inplace=True)
data.company.isnull().sum()


# In[59]:


plt.figure(figsize=(12,8))
data.company.value_counts()[:15].sort_values().plot(kind='barh',color='orchid')
plt.title("15 Companies with Highest Job Opportunities in India",size=18,weight='bold')
plt.xticks(size=11,weight='bold')
plt.yticks(size=11,weight='bold')
plt.show()


# Now we get 15 companies with highest job opportunities in India, three of them are Indian Institute of Technology Bombay, National Institute of Industrial Engineering, and Oracle India Pvt. Ltd.

# ## Most Needed Skill for Particular Industry ##

# In[60]:


# This section will help jop applicant to prepare their skills if they want to pursue career in specifit industry
mode_skills=[]
for industry in data.industry.unique():
    mode=data[data['industry']==industry][['skills','company']].groupby(['skills']).count().sort_values(by='company',ascending=False).index[0]
    mode_skills.append(mode)


# In[61]:


pd.set_option('max_rows',100)
pd.DataFrame(mode_skills,data.industry.unique().tolist(),columns=["Most Needed Skill"])[:63]


# ## Average Minimal Year Experience and Maximal Year Experience for Each Industry ##

# In[62]:


# This information can give insight for jobseeker to find a job that match their year experience
pd.set_option('max_rows',70)
display(data[['industry','min_year_exp','max_year_exp']].groupby(['industry']).mean().sort_values(by='min_year_exp').style.background_gradient(cmap='coolwarm'))


# From the result above, we find some industries that require relatively short year experience, such as Fresher/Trainee/Entry Level, Wellness/Fitness/Sports/ Beauty, and Leather. We also find some industries that require many years of experience, such as Tyres, Pulp and Paper, and Medical Devices / Equipments.

# ## Job Reccomendation ##

# In[63]:


# In this section, I want to make an interactive funtion to find jobs based on some criterias
# such as year experiece, degree, skill, and location


# In[64]:


@interact
def job_reccomendation(min_exp=range(0,25),max_exp=range(0,27),degree=list(data.degree.unique()),skills=data.skills.unique(),loc=list(x.unique())):
    bool_loc=[]
    for i in range(len(data)):
        if np.isin(loc,data['loc'][i]): bool_loc.append(bool(1))
        else : bool_loc.append(bool(0))
    y=data[(data["min_year_exp"]>=min_exp)&(data["max_year_exp"]<=max_exp)&(data["degree"]==degree)&(data["skills"]==skills)&(pd.Series(bool_loc))][['company','loc','industry', 'degree', 'experience','skills','numberofpositions','payrate' ]].reset_index(drop=True)
    return y


# In[ ]:




