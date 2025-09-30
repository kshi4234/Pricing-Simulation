# General Architecture

We want to be able to generate websites that will allow companies to visualize their margins, revenue, etc. Allow the AI agent access to the dataset (might take extremely long, so 
might not even do this).

We will have:
1. Agent to extract what metrics the user wants to model.
2. Agent to decide when more information is required from online
3. Agent to generate appropriate search queries, based on the type of product 
4. Simulation agent to properly model the elasticity of our product, using AI agent to perform our modeling (Not sure, this seems suspect)
5. Once elasticity modeled, no more agents, simply use rules to create interactive website with sliders and other such stuff. This part
can be vibe coded since I'm not good at front-end, just need to manually code the math going on behind the scenes.

                                User Prompt
                               /           \
                        