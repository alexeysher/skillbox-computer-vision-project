import streamlit as st

st.markdown('<h1 style="text-align:center; color:blue">Model selection</h2>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
c1.markdown('<h2 style="text-align:center">Base model</h3>', unsafe_allow_html=True)
c1.markdown("### :orange[1st type]: EfficientNetB0, input=224, feature size=1280")
c1.markdown("### :orange[2nd type]: EfficientNetB0, input=224, feature size=1280")
c2.markdown('<h2 style="text-align:center">On top model</h3>', unsafe_allow_html=True)
c2.markdown("### :orange[1st type]: (dropout=0.0, units=512), (dropout=0.0, units=256)")
c2.markdown("### :orange[2nd type]: (dropout=0.0, units=2048), (dropout=0.0, units=2048)")
st.markdown("---")
st.markdown('<h1 style="text-align:center; color:blue">Model learning</h2>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
c1.markdown('<h2 style="text-align:center">On top model learning</h3>', unsafe_allow_html=True)
c1.markdown("### :orange[1st type]: epoch=1, score=0.3402")
c1.markdown("### :orange[2nd type]: epoch=1, score=0.1658")
c2.markdown('<h2 style="text-align:center">Model fine tuning</h3>', unsafe_allow_html=True)
c2.markdown("### :orange[1st type]: epoch=22, score=0.519395")
c2.markdown("### :orange[2nd type]: epoch=31, score=0.3306")
st.markdown('---')
st.markdown('<h1 style="text-align:center; color:blue">Model testing</h2>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown('<h2 style="text-align:center">In process</h3>', unsafe_allow_html=True)
c1.image('https://user-images.githubusercontent.com/107345313/'
         '204358144-a0d4f282-84d9-4879-9b47-1fa3f4709131.png')
c2.markdown('<h2 style="text-align:center">At the end</h3>', unsafe_allow_html=True)
c2.image('https://user-images.githubusercontent.com/107345313/'
         '204358140-37ff0cf8-7f70-4a75-966b-1aa2aaf9300c.png')
c3.markdown('<h2 style="text-align:center">Results</h3>', unsafe_allow_html=True)
c3.image('https://user-images.githubusercontent.com/107345313/'
         '204360810-2e86e4c3-e5bd-4fbe-8dbf-5d9ecf614c8c.png')


