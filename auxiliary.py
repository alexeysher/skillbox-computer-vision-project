import streamlit as st
import streamlit.components.v1 as components
import textwrap


def wrap_text(text, length=50):
    '''
    Функция разбивает текст на строки заданной длины и заменяет символы переноса строки на HTML-элемент <br>

        Параметры:
        ----------
        text : string
            Обрабатываемый текст.

        length : int
            Максимальная длина строк

        Возвращаемый результат:
        -----------------------
            Объект типа string.
    '''
    return textwrap.fill(text, length).replace('\n', '<br>')


def set_text_style(text: str, tag: str = 'p', font_family: str = None, font_size: int = None,
                   color: str = None, background_color: str = None, text_align: str = None):
    variables = []
    if font_family is not None:
        variables.append(f'font-family: {font_family}')
    if font_size is not None:
        variables.append(f'font-size: {font_size}px')
    if color is not None:
        variables.append(f'color: {color}')
    if background_color is not None:
        variables.append(f'background-color: {background_color}')
    if text_align is not None:
        variables.append(f'text-align: {text_align}')
    variables.append('')
    style = '; '.join(variables)
    return f'<{tag} style="{style}">{text}</{tag}>'


def set_widget_style(widget_text, font_family: str = None, font_size: int = None,
                     color: str = None, background_color: str = None, text_align: str = None):
    html = \
        """
    <script>
        var elements = window.parent.document.querySelectorAll('*'), i;
        for (i = 0; i < elements.length; ++i) { 
            if (elements[i].innerText == '""" + widget_text + "') {"
    if font_family is not None:
        html += \
            f"""
                elements[i].style.fontFamily='{font_family}';
        """
    if font_size is not None:
        html += \
            f"""
                elements[i].style.fontSize='{font_size}px';
        """
    if color is not None:
        html += \
            f"""
                elements[i].style.color='{color}';
        """
    if background_color is not None:
        html += \
            f"""
                elements[i].style.backgroundColor='{background_color}';
        """
    if text_align is not None:
        html += \
            f"""
                elements[i].style.textAlign='{text_align}';
        """
    html += \
        """
            } 
        } 
    </script> 
    """
    components.html(html, height=0, width=0)


def hide_menu_button():
    """
    Скрывает кнопку вызова меню.
    """
    st.markdown(
        """
        <style>
            MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )


def remove_blank_space():
    """
    Удаляет пустое пространство вверху страницы.
    """
    st.markdown(f'''
                <style>
                    .css-k1ih3n {{
                        padding-top: 1.5rem;
                    }}
                </style>
                <style>
                    .css-1vq4p4l {{
                        padding-top: 4.0rem;
                    }}
                </style>
                ''', unsafe_allow_html=True,
                )
