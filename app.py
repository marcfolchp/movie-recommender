import streamlit as st
from recommender_models import get_top_non_personalized, recommend_user_based, recommend_item_based, recommend_content_based, recommend_genai, recommend_svd_sklearn, recommend_hybrid

class PageRoutes:
    HOME = "Home"
    NON_PERSONALISED = "Non-personalised"
    USER_BASED_CF = "User-based Collaborative Filtering"
    CONTENT_BASED = "Content-based Filtering"
    MATRIX_FACT = "Matrix Factorisation"
    HYBRID = "Hybrid Approach"
    GENERATIVE_AI = "Generative AI"

def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()

def layout_home_buttons():
    buttons = [
        (PageRoutes.NON_PERSONALISED, "Non-personalised"),
        (PageRoutes.USER_BASED_CF, "Collaborative Filtering"),
        (PageRoutes.CONTENT_BASED, "Content-based Filtering"),
        (PageRoutes.MATRIX_FACT, "Matrix Factorisation"),
        (PageRoutes.HYBRID, "Hybrid Approach"),
        (PageRoutes.GENERATIVE_AI, "Generative AI")
    ]

    col1, col2 = st.columns(2)
    for idx, (route, label) in enumerate(buttons):
        col = col1 if idx % 2 == 0 else col2
        with col:
            if st.button(label, use_container_width=True):
                set_page(route)

def layout_header(title):
    st.markdown(
        f"""
        <div style="text-align:center; margin-top: 2rem;">
            <h1 style="margin-bottom:0.25rem;">Movie Recommender System</h1>
            <h3 style="color:gray; font-weight:normal;">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

def home_page(_=None):
    layout_header("Select a Recommendation Model")
    st.markdown("<br>", unsafe_allow_html=True)
    layout_home_buttons()

def placeholder_page(title):
    layout_header(title)
    st.info("This page is under development.")
    if st.button("Back to Home", use_container_width=True):
        set_page(PageRoutes.HOME)

def navigate():
    page = st.session_state.get("page", PageRoutes.HOME)
    pages = {
        PageRoutes.HOME: home_page,
        PageRoutes.NON_PERSONALISED: non_personalized_page,
        PageRoutes.USER_BASED_CF: collaborative_filtering_page,
        PageRoutes.CONTENT_BASED: content_based_page,
        PageRoutes.MATRIX_FACT: matrix_factorisation_page,
        PageRoutes.HYBRID: hybrid_approach_page,
        PageRoutes.GENERATIVE_AI: generative_ai_page
    }
    pages.get(page, home_page)(page)

def non_personalized_page(title):
    layout_header(title)
    
    st.markdown(
        """
        <p style='text-align:center; color:gray; font-size:1.05rem;'>
        These are general-purpose recommendations that do not depend on individual user preferences.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("non_personalized_form"):
        method = st.selectbox(
            "Select a method:",
            options=["editorial", "top_n", "average_rating", "weighted", "association"]
        )
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        df = get_top_non_personalized(method)
        st.dataframe(df, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back to Home", use_container_width=True):
        set_page(PageRoutes.HOME)

def collaborative_filtering_page(title):
    layout_header(title)

    st.markdown(
        """
        <p style='text-align:center; color:gray; font-size:1.05rem;'>
        Compare two collaborative filtering strategies: one based on users with similar preferences,<br>
        and the other on items with similar rating patterns.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("user_input_form"):
        user_id = st.text_input("Enter your User ID", value="", max_chars=10)
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        if not user_id.strip().isdigit():
            st.error("Please enter a valid numeric User ID.")
            return

        user_id = int(user_id.strip())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("User-Based")
            df_user = recommend_user_based(user_id)
            st.dataframe(df_user, use_container_width=True)

        with col2:
            st.subheader("Item-Based")
            df_item = recommend_item_based(user_id)
            st.dataframe(df_item, use_container_width=True)

        common_movies = set(df_user["movie_title"]).intersection(df_item["movie_title"])
        st.markdown(f"<br><b>Number of movies in both lists: {len(common_movies)}</b>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back to Home", use_container_width=True):
        set_page(PageRoutes.HOME)

def content_based_page(title):
    layout_header(title)

    st.markdown(
        """
        <p style='text-align:center; color:gray; font-size:1.05rem;'>
        This approach recommends movies based on a user's preferences inferred from the content of items they’ve liked.

        </p>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("content_input_form"):
        user_id = st.text_input("Enter your User ID", value="", max_chars=10)
        submitted = st.form_submit_button("Get Content-Based Recommendations")

    if submitted:
        if not user_id.strip().isdigit():
            st.error("Please enter a valid numeric User ID.")
            return

        user_id = int(user_id.strip())
        df_content = recommend_content_based(user_id)
        st.dataframe(df_content, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back to Home", use_container_width=True):
        set_page(PageRoutes.HOME)

def matrix_factorisation_page(title):
    layout_header(title)

    st.markdown(
        """
        <p style='text-align:center; color:gray; font-size:1.05rem;'>
        Matrix Factorisation uses latent features to identify hidden patterns in user-item interactions.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("svd_input_form"):
        user_id = st.text_input("Enter your User ID", value="", max_chars=10)
        submitted = st.form_submit_button("Get Matrix Factorisation Recommendations")

    if submitted:
        if not user_id.strip().isdigit():
            st.error("Please enter a valid numeric User ID.")
            return

        user_id = int(user_id.strip())
        df_svd, rmse = recommend_svd_sklearn(user_id=user_id, num_recommendations=10)

        st.dataframe(df_svd, use_container_width=True)

        st.markdown(
            f"<br><b>The RMSE of this model is: {rmse:.4f}</b>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back to Home", use_container_width=True):
        set_page(PageRoutes.HOME)

def hybrid_approach_page(title):
    layout_header(title)

    st.markdown(
        """
        <p style='text-align:center; color:gray; font-size:1.05rem;'>
        This hybrid model blends content-based and collaborative filtering techniques.<br>
        It balances personalization and relevance using a tunable parameter.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("hybrid_input_form"):
        user_id = st.text_input("Enter your User ID", value="", max_chars=10)
        submitted = st.form_submit_button("Get Hybrid Recommendations")

    if submitted:
        if not user_id.strip().isdigit():
            st.error("Please enter a valid numeric User ID.")
            return

        user_id = int(user_id.strip())

        df_hybrid = recommend_hybrid(user_id=user_id, num_recommendations=10, alpha=0.5)
        st.dataframe(df_hybrid, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back to Home", use_container_width=True):
        set_page(PageRoutes.HOME)

def generative_ai_page(title):
    layout_header(title)

    st.markdown(
        """
        <p style='text-align:center; color:gray; font-size:1.05rem;'>
        This model leverages Generative AI to understand natural language descriptions<br>
        and suggest relevant movies.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("genai_input_form"):
        query = st.text_area("Describe what kind of movies you're looking for", height=100)
        submitted = st.form_submit_button("Generate Recommendations")

    if submitted:
        if not query.strip():
            st.error("Please enter a descriptive query.")
            return

        df_genai = recommend_genai(query)
        st.dataframe(df_genai, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Back to Home", use_container_width=True):
        set_page(PageRoutes.HOME)

if __name__ == "__main__":
    navigate()