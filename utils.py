from umap.umap_ import UMAP
import altair as alt

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def sc_plot(df, size=60, color='steelblue'):
    cols = list(df.columns)
    chart = alt.Chart(df).mark_circle(size=size, color=color).encode(
        x=alt.X('x',
              scale=alt.Scale(zero=False)
              ),
        y=alt.Y('y',
                scale=alt.Scale(zero=False)
                ),
        tooltip=cols
    ).properties(
        width=700,
        height=400
    )
    
    return chart
    

def umap_reduct(emb):
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = UMAP(n_neighbors=100)
    umap_embeds = reducer.fit_transform(emb)
    return umap_embeds