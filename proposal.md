# Team Name
Dr. Cusack
Scheduling Advisor

# Problem / Domain

Targeted towards first and second year students who are still figuring out which courses to take and explore possibilities in a Computer Science Major. The LLM should be able to provide specific advice to the student about courses to take towards Hope's Anchor Plan as well as towards their desired major. 


# Planned Features
- provide information that can be found on Degree Works as well as the Schedule Planner in one place
- provide specific advice to the student about courses to take towards Hope's Anchor Plan as well as towards their desired major
- give explanations about how each course is relevant in the real world
- give examples of how their major can be applied after college
- 

# Technical Approach
- Course descriptions stored in documents made available to the LLM
- Hope's Anchor Plan requirements stored in documents (or it can search it up if we figure out how to do that)
- Streamlit web app: text input asking for a link to a major's course requirements, e.g B.A. in Computer Science: https://catalog.hope.edu/preview_program.php?catoid=8&poid=1781
    it would be nice to have it use its own general knowledge to give examples of how the major is used in the working world
- Tool: search_docs to search documents available

# Team Member Roles

Alem:
- Web Scraping Tool
- Conditional Workflow for Web Scraping Tool
- TypeDict State
- Live Demo

Eleanor:
- Add Course Descriptions
- Ensure search_docs tool works
- Adding the Course Schedules
- Scheduling Tool
- Streamlit Initialization 
- Readme

Katie:
- Think of possible graph structures to incoporate conditional routing 
- Add URLs of the course requirements for supported majors
- Major URL Lookup Table
- Major URL Lookup Tool
- Anchor Plan Requirements


All three of us do testing


# Rough Timeline
March 17
Come up nodes work on graphs, think if we need more tools

Week 11
Graph mostly done

Week 12
Halfway to working prototype

Week 13
WebApp work

Week 14
Working prototype + adding features
Work on presentation

Week 15
Presentation complete
App complete (Hopefully!)

