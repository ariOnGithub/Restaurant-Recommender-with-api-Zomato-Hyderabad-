import streamlit as st
import requests

def recommend(name):
    resp = requests.get('http://localhost:8000/restaurants/' + str(name))
    return resp.json()


def main():
    st.title('Restaurant Recommender')
    name = st.text_input('Enter restaurant name: ')

    if st.button('Recommendations'):
        output = recommend(name)
        col0, col1, col2, col3 = st.columns(4)
        print(output)
        with col0:
            st.header('Name')
            for i in output['Cuisines']:
                st.write(i)
        with col1:
            st.header("Cuisines")
            for i in output['Cuisines'].values():
                st.write(i)
        with col2:
            st.header("Rating")
            for i in output['Mean Rating'].values():
                st.write(i)
        with col3:
            st.header("Cost")
            for i in output['Cost'].values():
                st.write(i)

if __name__ == '__main__':
    main()

